// The demo's solver: cuwave's 2D scalar kernels driven as simulate drives them.
// module.cpp is what cpu.RawModule compiles (cpu.hpp, the stencil table,
// common.cuh, scalar.cu), written by build.sh. The grid is the visible nodes
// plus a margin of `width` nodes on every side, which per face is outside the
// domain (Neumann), outside behind held nodes (Dirichlet) or a sponge
// (absorbing): Simulation.domain, .dirichlet and boundary.sponge.
// Compile-time configuration (set via -D flags in build.sh):
//   NDIM = 2, USE_FLOAT, USE_DAMPING, USE_DOMAIN, DIRICHLET_FACES = 0
//   CUWAVE_THREADS: the step's blocks across a pool of threads

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>
#ifdef CUWAVE_THREADS
#include <atomic>
#include <thread>
#ifdef __EMSCRIPTEN__
#include <emscripten/threading.h>
#endif
#endif

#include "module.cpp" // after the library, whose __noinline__ it redefines

namespace {

constexpr int ROWS = 8, COLS = 32; // threads per block, axis 0 and axis 1
constexpr int DEEP = 1 << 30;      // a node whose stencil never reaches a wall
constexpr int REFILL = 256;        // threads per block of the Neumann refill
constexpr int SOURCES = 16;        // bursts sounding at once, the oldest dropped
constexpr double PI = 3.14159265358979323846;
enum Mode { NEUMANN, DIRICHLET_WALL, ABSORBING };

struct Burst {
  int node;
  int start; // the step its signal starts at
  double amplitude, frequency;
  int cycles;
};

struct Source {
  int node;
  float value; // weight times signal, as excitation_kernel adds it
};

// settings
float courant = 0.5f, decay = 0, beta = 0;
int width = 1;
int modes[4] = {NEUMANN, NEUMANN, NEUMANN, ABSORBING}; // per 2 * axis + side

// grid, ghosts included: rows `n0`, columns `n1`, margins `margin` wide
int n0 = 0, n1 = 0, margin = 0, t = 0;
std::vector<float> U[2], stiff, damping, weight;
std::vector<unsigned char> solid, stepped, held;
std::vector<int> cells, tiles;
std::vector<Burst> bursts;
std::vector<Source> schedule;

// excitation scratch, refilled per step
std::vector<int> nodes;
std::vector<float> signal, weights;

// ------------------------------------------- threads
#ifdef CUWAVE_THREADS
constexpr int SPIN = 1 << 16; // polls before a worker sleeps, past a frame's steps
constexpr int FIELD = 20;     // bits of the block count and of the next block
constexpr uint64_t MASK = (uint64_t(1) << FIELD) - 1;
int workers = 0; // detached, so they end with the process
std::atomic<int> generation{0};
std::atomic<uint64_t> cursor{0}; // epoch, block count, next block: one word
std::atomic<long> done{0};
std::function<void(long)> job;

void sleep_on(std::atomic<int> &word, const int seen) {
#ifdef __EMSCRIPTEN__
  emscripten_futex_wait(&word, seen, INFINITY);
#else
  word.wait(seen);
#endif
}

void wake_all(std::atomic<int> &word) {
#ifdef __EMSCRIPTEN__
  emscripten_futex_wake(&word, INT32_MAX);
#else
  word.notify_all();
#endif
}

// the blocks left of the current job, one at a time; a block is claimed with
// its job's epoch, and `parallel` waits for every claimed one, so `job` holds
void drain() {
  for (uint64_t v = cursor.load();;) {
    const long b = long(v & MASK), count = long(v >> FIELD & MASK);
    if (b >= count)
      return;
    if (!cursor.compare_exchange_weak(v, v + 1))
      continue; // v reloaded
    job(b);
    done.fetch_add(1);
    v = cursor.load();
  }
}

// a worker from generation `seen` on, which it has already run
void work(int seen) {
  for (;;) {
    for (int k = 0; k < SPIN && generation.load() == seen; ++k)
      ;
    while (generation.load() == seen)
      sleep_on(generation, seen);
    seen = generation.load();
    drain();
  }
}

// `count` blocks of `run` across this thread and the workers awake, returning
// when all are done, so a worker still asleep holds no step up
void parallel(const long count, std::function<void(long)> run) {
  if (count > long(MASK)) { // past the packed count: this thread alone
    for (long b = 0; b < count; ++b)
      run(b);
    return;
  }
  job = std::move(run);
  done = 0;
  const uint64_t epoch = (cursor.load() >> 2 * FIELD) + 1;
  cursor = epoch << 2 * FIELD | uint64_t(count) << FIELD;
  generation.fetch_add(1);
  wake_all(generation);
  drain();
  while (done.load() < count)
    ;
}
#endif

// cuwave_launch of cpu.hpp, its blocks across the threads
template <class... A>
void launch(void (*kernel)(A...), const dim3 grid, const dim3 block,
            void **args) {
#ifdef CUWAVE_THREADS
  if (workers) {
    gridDim = grid;
    blockDim = block;
    parallel((long)grid.x * grid.y * grid.z, [&](const long b) {
      blockIdx = dim3{unsigned(b % grid.x), unsigned(b / grid.x % grid.y),
                      unsigned(b / ((long)grid.x * grid.y))};
      for (unsigned z = 0; z < block.z; ++z)
        for (unsigned y = 0; y < block.y; ++y)
          for (unsigned x = 0; x < block.x; ++x) {
            threadIdx = dim3{x, y, z};
            cuwave_call(kernel, args, std::index_sequence_for<A...>{});
          }
    });
    return;
  }
#endif
  cuwave_launch(kernel, grid, block, args);
}

// ------------------------------------- discretization setup
// whether interior node (a0, a1) lies in the margin of face `f`
bool in_margin(const int f, const int a0, const int a1) {
  const int a = f < 2 ? a0 : a1, n = f < 2 ? n0 : n1;
  return f % 2 ? a >= n - 1 - margin : a <= margin;
}

// the nodes the step updates (Simulation.domain less .dirichlet) and the held
// nodes (Simulation.dirichlet) of rows `first` to `last`, from the walls and
// the face modes, the field zeroed off the domain
void build_masks(const int first, const int last) {
  for (int a0 = std::max(1, first); a0 < std::min(n0 - 1, last); ++a0)
    for (int a1 = 1; a1 < n1 - 1; ++a1) {
      bool open = !solid[a0 * n1 + a1], dirichlet = false;
      for (int f = 0; f < 4; ++f)
        if (in_margin(f, a0, a1)) {
          open = open && modes[f] == ABSORBING;
          dirichlet = dirichlet || modes[f] == DIRICHLET_WALL;
        }
      stepped[a0 * n1 + a1] = open;
      held[a0 * n1 + a1] = !open && dirichlet;
      if (!open)
        U[0][a0 * n1 + a1] = U[1][a0 * n1 + a1] = 0;
    }
}

// domain_cells of wave.py: per node of rows `first` to `last` the radii of its
// two cells along each axis
void build_cells(const int first, const int last) {
  const int R = STENCIL_RADIUS, n[2] = {n0, n1};
  const auto interior = [&](const int a0, const int a1) {
    return a0 > 0 && a0 < n0 - 1 && a1 > 0 && a1 < n1 - 1;
  };
  const auto inside = [&](const int a0, const int a1) {
    return !interior(a0, a1) || stepped[a0 * n1 + a1]; // ghosts count
  };
  const auto holds = [&](const int a0, const int a1) {
    return a0 >= 0 && a0 < n0 && a1 >= 0 && a1 < n1 && held[a0 * n1 + a1];
  };
  for (int a0 = std::max(0, first); a0 < std::min(n0, last); ++a0)
    for (int a1 = 0; a1 < n1; ++a1) {
      cells[a0 * n1 + a1] = 0;
      if (!inside(a0, a1))
        continue;
      const int a[2] = {a0, a1};
      int code = 0;
      bool deep = true;
      for (int d = 0; d < 2; ++d) {
        const int wall = std::min(R, std::min(a[d], n[d] - 1 - a[d]));
        const auto along = [&](const int k, int &b0, int &b1) {
          b0 = d == 0 ? a0 + k : a0, b1 = d == 1 ? a1 + k : a1;
        };
        // the run of stepped nodes on one side, capped by the wall
        const auto run = [&](const int sign) {
          int count = 0, b0, b1;
          bool on = true;
          for (int k = 1; k <= R; ++k) {
            along(sign * k, b0, b1);
            const int b = a[d] + sign * k;
            on = on && (b < 0 || b >= n[d] || inside(b0, b1));
            count += on && k <= wall;
          }
          return count;
        };
        const int up = run(1), down = run(-1);
        int rp = std::min(wall, std::min(1 + down, up));
        int rm = std::min(wall, std::min(down, 1 + up));
        int b0, b1;
        along(1, b0, b1);
        rp = rp == 0 && holds(b0, b1) ? DIRICHLET : rp;
        along(-1, b0, b1);
        rm = rm == 0 && holds(b0, b1) ? DIRICHLET : rm;
        code |= (rp | rm << 3) << (6 * d);
        deep = deep && rp == wall && rm == wall;
      }
      cells[a0 * n1 + a1] = deep ? DEEP : code;
    }
}

// domain_tiles of wave.py: the tiles holding a stepped node, bit 30 at a wall
void build_tiles() {
  tiles.clear();
  const int g0 = (n0 + ROWS - 1) / ROWS, g1 = (n1 + COLS - 1) / COLS;
  for (int b0 = 0; b0 < g0; ++b0)
    for (int b1 = 0; b1 < g1; ++b1) {
      bool occupied = false, wall = false;
      for (int a0 = std::max(1, b0 * ROWS);
           a0 < std::min(n0 - 1, (b0 + 1) * ROWS); ++a0)
        for (int a1 = std::max(1, b1 * COLS);
             a1 < std::min(n1 - 1, (b1 + 1) * COLS); ++a1) {
          const int code = cells[a0 * n1 + a1];
          occupied = occupied || code != 0;
          wall = wall || code != DEEP;
        }
      if (occupied)
        tiles.push_back(int(wall) << 30 | b0 << 10 | b1);
    }
}

// boundary.sponge behind the absorbing faces plus a uniform decay, and the
// excitation weight dt**2 / (1 + beta) the damped update divides by
void build_damping() {
  const float dt = courant;
  damping.assign(n0 * n1, 0);
  weight.assign(n0 * n1, 0);
  for (int a0 = 1; a0 < n0 - 1; ++a0)
    for (int a1 = 1; a1 < n1 - 1; ++a1) {
      float taper = 0;
      for (int f = 0; f < 4; ++f) {
        if (modes[f] != ABSORBING)
          continue;
        const int a = f < 2 ? a0 : a1, n = f < 2 ? n0 : n1;
        const float wall = float(f % 2 ? n - 2 : 1);
        const float ramp = 1.f - std::fabs(float(a) - wall) / float(margin);
        taper = std::max(taper, std::min(1.f, std::max(0.f, ramp)));
      }
      const float sponge = float(2.0 * beta) * (taper * taper) / dt;
      const float d = decay + sponge;
      damping[a0 * n1 + a1] = d;
      weight[a0 * n1 + a1] =
          float(double(dt) * dt) * (1.f / (1.f + 0.5f * d * dt));
    }
}

// signals.sineburst
float sineburst(const double time, const Burst &b) {
  if (!(time > 0 && time <= b.cycles / b.frequency))
    return 0;
  const double window = std::sin(PI * b.frequency * time / b.cycles);
  return float(b.amplitude * std::sin(2 * PI * b.frequency * time) *
               (window * window));
}

// ----------------------------------- simulation functions
// the bursts sounding at step t on stepped nodes, the rest dropped
void sources() {
  nodes.clear();
  signal.clear();
  weights.clear();
  for (size_t i = 0; i < bursts.size();) {
    const Burst &burst = bursts[i];
    const double time = double(courant) * (t - burst.start);
    if (time > burst.cycles / burst.frequency || !cells[burst.node]) {
      bursts.erase(bursts.begin() + i);
      continue;
    }
    nodes.push_back(burst.node);
    signal.push_back(sineburst(time, burst));
    weights.push_back(weight[burst.node]);
    ++i;
  }
}

// one step of simulate: the step, the excitation, the Neumann refill
void step() {
  float *u0 = U[t % 2].data(), *u1 = U[1 - t % 2].data(), *u2 = u0;
  const float *k = stiff.data(), *d = damping.data();
  const int *c = cells.data(), *b = tiles.data();
  int derive = 1, N0 = n0, N1 = n1, s0 = n1;
  float dt = courant, f = float(2.0 * double(courant) * courant);
  void *args[] = {&u0, &u1, &u2, &k, &k, &derive, &d,  &dt,
                  &c,  &b,  &f,  &N0, &f, &N1, &s0};
  launch(fd_kernel, dim3{unsigned(tiles.size())}, dim3{COLS, ROWS}, args);

  sources();
  if (!nodes.empty()) {
    const float *source = signal.data(), *w = weights.data();
    const int *index = nodes.data();
    int offset = 0, num = int(nodes.size());
    void *excite[] = {&u2, &source, &offset, &index, &num, &w};
    cuwave_launch(excitation_kernel, dim3{unsigned(num + REFILL - 1) / REFILL},
                  dim3{REFILL}, excite);
  }

  int faces = 0b1111, ghosts = 2 * (n0 - 2) + 2 * (n1 - 2);
  void *refill[] = {&u2, &faces, &N0, &N1, &s0};
  cuwave_launch(homogeneous_neumann_kernel,
                dim3{unsigned(ghosts + REFILL - 1) / REFILL}, dim3{REFILL},
                refill);
  ++t;
}

} // namespace

// ------------------------------------------ exports
extern "C" {

// the settings `resize` builds the damping from: Courant number c dt / dx,
// uniform damping `d`, and the margins' `w` nodes, a sponge ramping to `b`
void configure(const float c, const float d, const int w, const float b) {
  courant = c, decay = d, width = w, beta = b;
}

// face 2 * axis + side (top, bottom, left, right) set to `mode`: 0 Neumann,
// 1 Dirichlet, 2 absorbing; a caller setting them calls `rebuild`
void boundary(const int face, const int mode) { modes[face] = mode; }

// the masks, cells, tiles and damping of the walls and faces, the field
// zeroed off the domain
void rebuild() {
  stepped.assign(n0 * n1, 0);
  held.assign(n0 * n1, 0);
  cells.assign(n0 * n1, 0);
  build_masks(0, n0);
  build_damping();
  build_cells(0, n0);
  build_tiles();
}

// `rebuild` after walls changed in visible rows `first` to `last` only
void repaint(const int first, const int last) {
  const int a = first + margin + 1, b = last + margin + 1;
  build_masks(a, b);
  build_cells(a - STENCIL_RADIUS, b + STENCIL_RADIUS);
  build_tiles();
}

// the flat index of visible node (0, 0)
int origin() { return (margin + 1) * n1 + margin + 1; }

// a grid of `rows` by `cols` visible nodes, each `scale` old nodes wide, the
// walls and field sampled at the old node under each new one's center; a new
// step spans `scale` old ones, so the earlier level keeps the time derivative
void resize(const int rows, const int cols, const double scale) {
  const int m1 = n1, from = origin(), m0 = n0 - 2 * margin - 2,
            mc = n1 - 2 * margin - 2;
  std::vector<float> old[2] = {U[(t + 1) % 2], U[t % 2]};
  std::vector<unsigned char> walls = solid;
  margin = width, n0 = rows + 2 * margin + 2, n1 = cols + 2 * margin + 2;
  for (auto &u : U)
    u.assign(n0 * n1, 0);
  solid.assign(n0 * n1, 0);
  stiff.assign(n0 * n1, 1);
  const int to = origin();
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      const int a = int((r + 0.5) * scale), b = int((c + 0.5) * scale);
      if (a >= m0 || b >= mc)
        continue;
      const int i = from + a * m1 + b, j = to + r * n1 + c;
      solid[j] = walls[i];
      U[1][j] = old[0][i]; // the latest level
      U[0][j] = scale == 1 ? old[1][i]
                           : float(old[0][i] - scale * (old[0][i] - old[1][i]));
    }
  t = 0, bursts.clear();
  rebuild();
}

// the grid's row stride and row count, ghosts and margins included
int stride() { return n1; }
int extent() { return n0; }

// the latest field level, `stride()` floats a row
float *field() { return U[(t + 1) % 2].data(); }

// the walls, one byte a node; a caller writing them calls `rebuild`
unsigned char *walls() { return solid.data(); }

// the step's cell codes and damping, for a GPU stepping the same grid
int *codes() { return cells.data(); }
float *dampings() { return damping.data(); }

// the stencil table row r, entry k (1-indexed), and the radius
float coefficient(const int r, const int k) { return OP_W(r, k); }
int radius() { return STENCIL_RADIUS; }

// a sine burst at visible node (`row`, `col`), from the next step on
void burst(const int row, const int col, const double amplitude,
           const double frequency, const int cycles) {
  const int node = origin() + row * n1 + col;
  if (row < 0 || row >= n0 - 2 * margin - 2 || col < 0 ||
      col >= n1 - 2 * margin - 2 || !cells[node])
    return;
  if (bursts.size() == SOURCES)
    bursts.erase(bursts.begin());
  bursts.push_back({node, t, amplitude, frequency, cycles});
}

void advance(const int steps) {
  for (int k = 0; k < steps; ++k)
    step();
}

// the sources of the next `steps` steps, SOURCES a step (node -1 unused),
// the clock advanced past them, for a GPU stepping the field
Source *plan(const int steps) {
  schedule.assign(steps * SOURCES, {-1, 0.f});
  for (int k = 0; k < steps; ++k, ++t) {
    sources();
    for (size_t i = 0; i < nodes.size(); ++i)
      schedule[k * SOURCES + i] = {nodes[i], weights[i] * signal[i]};
  }
  return schedule.data();
}

// `count` threads step the field, this one included; 1 without CUWAVE_THREADS
int threads(const int count) {
#ifdef CUWAVE_THREADS
  for (; workers < count - 1; ++workers)
    std::thread(work, generation.load()).detach();
  return workers + 1;
#else
  return 1;
#endif
}

// every field level and burst zeroed, the walls kept
void clear() {
  for (auto &u : U)
    std::fill(u.begin(), u.end(), 0.f);
  bursts.clear();
}

} // extern "C"
