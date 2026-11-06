// Prelude prepended to every .cu by compile_kernels, after the stencil table.
// Compile-time configuration (set via -D flags and stencils.preamble):
//   USE_FLOAT
//   NDIM = 1 | 2 | 3
//   STENCIL_RADIUS
//   OP_COEFFS
//   STAG_COEFFS
//   NODE_CLOSURE, HALF_CLOSURE, NODE_SPAN, HALF_SPAN, NODE_ROWS_C, HALF_ROWS_C
//   DIRICHLET_FACES (default 0)

#ifdef USE_FLOAT
typedef float real_t;
#define MUL_RN __fmul_rn // a product nvcc never fuses into a sum
#define FMA_RN __fmaf_rn
#else
typedef double real_t;
#define MUL_RN __dmul_rn // pinned too, so code around it rounds nothing anew
#define FMA_RN __fma_rn
#endif

// ----------------------------- finite difference helpers
#ifndef STENCIL_RADIUS
#define STENCIL_RADIUS 1 // default order 2
#endif

#if STENCIL_RADIUS == 1
#define OP_W(r, k) ((real_t)1)
#define SG_W(r, k) ((real_t)1)
#else
// cache
__constant__ real_t OP_C[STENCIL_RADIUS][STENCIL_RADIUS] = OP_COEFFS;
#define OP_W(r, k) OP_C[(r) - 1][(k) - 1] // 1-indexed adjustment
__constant__ real_t SG_C[STENCIL_RADIUS][STENCIL_RADIUS] = STAG_COEFFS;
#define SG_W(r, k) SG_C[(r) - 1][(k) - 1] // 1-indexed adjustment
#endif

// ---------------------------------- mirrored taps
#ifndef DIRICHLET_FACES
#define DIRICHLET_FACES 0
#endif

// whether node a is the wall node of a face of `odd` (bit 0 low, bit 1 high)
__device__ __forceinline__ bool held_wall(const int a, const int N,
                                          const int odd) {
  return (odd & 1 && a == 1) || (odd & 2 && a == N - 2);
}

// node b of an axis of N nodes, past the stored ghost its mirror about the
// wall node, odd past a face of `odd`
__device__ __forceinline__ int mirror_node(const int b, const int N,
                                           const int odd, int &sign) {
  sign = (b < 0 && odd & 1) || (b > N - 1 && odd & 2) ? -1 : 1;
  return b < 0 ? 2 - b : b > N - 1 ? 2 * (N - 2) - b : b;
}

// a value read off a mirror with its sign
__device__ __forceinline__ real_t signed_value(const real_t v, const int sign) {
  return sign > 0 ? v : sign < 0 ? -v : (real_t)0;
}

// u at node a + j of the line through idx, read through mirror_node
__device__ __forceinline__ real_t mirrored(const real_t *__restrict__ u,
                                           const int idx, const int s,
                                           const int a, const int N,
                                           const int odd, const int j) {
  int sign;
  const int b = mirror_node(a + j, N, odd, sign);
  return signed_value(u[idx + (b - a) * s], sign);
}

// node b past the wall node its image, for a field no refill keeps beyond it:
// odd across a face of `odd`
__device__ __forceinline__ int node_image(const int b, const int N,
                                          const int odd, int &sign) {
  sign = (b < 1 && odd & 1) || (b > N - 2 && odd & 2) ? -1 : 1;
  return b < 1 ? 2 - b : b > N - 2 ? 2 * (N - 2) - b : b;
}

// half point b, half a node up from node b, past the wall its image
__device__ __forceinline__ int half_image(const int b, const int N,
                                          const int odd, int &sign) {
  sign = (b < 1 && odd & 1) || (b > N - 3 && odd & 2) ? -1 : 1;
  return b < 1 ? 1 - b : b > N - 3 ? 2 * N - 5 - b : b;
}

// a node-centred field at a + j of the line through idx, through node_image
__device__ __forceinline__ real_t node_imaged(const real_t *__restrict__ u,
                                              const int idx, const int s,
                                              const int a, const int N,
                                              const int odd, const int j) {
  int sign;
  const int b = node_image(a + j, N, odd, sign);
  return signed_value(u[idx + (b - a) * s], sign);
}

// a half-point field at a + j of the line through idx, through half_image
__device__ __forceinline__ real_t half_imaged(const real_t *__restrict__ u,
                                              const int idx, const int s,
                                              const int a, const int N,
                                              const int odd, const int j) {
  int sign;
  const int b = half_image(a + j, N, odd, sign);
  return signed_value(u[idx + (b - a) * s], sign);
}

// t[j - first] = a field at a + j, j = first .. first + 2R - 1, read in place
// where every tap lies on stored points `lo .. hi`, else through its image
template <bool HALF>
__device__ __forceinline__ void staged_taps(real_t *t,
                                            const real_t *__restrict__ u,
                                            const int idx, const int s,
                                            const int a, const int N,
                                            const int odd, const int first,
                                            const int lo, const int hi) {
  if (a + first >= lo && a + first + 2 * STENCIL_RADIUS - 1 <= hi) {
#pragma unroll
    for (int j = 0; j < 2 * STENCIL_RADIUS; ++j)
      t[j] = u[idx + (first + j) * s];
  } else {
#pragma unroll
    for (int j = 0; j < 2 * STENCIL_RADIUS; ++j)
      t[j] = HALF ? half_imaged(u, idx, s, a, N, odd, first + j)
                  : node_imaged(u, idx, s, a, N, odd, first + j);
  }
}

// --------------------------------- staggered lattice
#define NPAIRS (NDIM * (NDIM - 1) / 2)
#define NVOIGT (NDIM + NPAIRS)
#if NDIM == 3
#define PAIR_ROW(k, l) (NDIM + 3 - (k) - (l))
#else
#define PAIR_ROW(k, l) 2
#endif

// radius of a half-point derivative, whose taps may sit on the wall
__device__ __forceinline__ int rad_half(const int a, const int N) {
  return min(STENCIL_RADIUS, min(a, N - 2 - a));
}

// the node whose tile writes `a` back in a blocked step: a ghost goes with its
// mirror, the one tile where the mirror stays valid
__device__ __forceinline__ int owner(const int a, const int N) {
  return a == 0 ? 2 : a == N - 1 ? N - 3 : a;
}

__device__ __forceinline__ bool clamped_face(const int clamped, const int d,
                                             const int side) {
  return (clamped >> (2 * d + side)) & 1;
}

// ---------------------------------- staggered closure
#ifdef NODE_CLOSURE
// the low wall's rows of the staggered SBP pair, row 0 the wall's; the high
// wall reads them mirrored and negated
__device__ const real_t NODE_ROWS[NODE_CLOSURE][NODE_SPAN] = NODE_ROWS_C;
__device__ const real_t HALF_ROWS[HALF_CLOSURE][HALF_SPAN] = HALF_ROWS_C;
#define CLOSURE_BAND                                                           \
  (NODE_CLOSURE > HALF_CLOSURE ? NODE_CLOSURE : HALF_CLOSURE)

// a node past every wall's closure band: the interior rows on every axis
__device__ __forceinline__ bool deep_node(const int *A, const int *NN) {
  bool deep = true;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    deep &= A[d] > CLOSURE_BAND && A[d] < NN[d] - 2 - CLOSURE_BAND;
  return deep;
}

// a line through a field: u(j) the value j points along the axis from idx
struct Line {
  const real_t *__restrict__ p;
  int idx, s;
  __device__ real_t operator()(const int j) const { return p[idx + j * s]; }
};

// the closure rows off either wall
template <class L>
__device__ __forceinline__ real_t node_rows(const L &u, const int lo,
                                           const int hi) {
  real_t acc = (real_t)0;
  if (lo < NODE_CLOSURE) {
#pragma unroll
    for (int m = 0; m < NODE_SPAN; ++m)
      acc += NODE_ROWS[lo][m] * u(m - lo);
  } else {
#pragma unroll
    for (int m = 0; m < NODE_SPAN; ++m)
      acc -= NODE_ROWS[hi][m] * u(hi - 1 - m);
  }
  return acc;
}

template <class L>
__device__ __forceinline__ real_t half_rows(const L &u, const int lo,
                                           const int hi) {
  real_t acc = (real_t)0;
  if (lo < HALF_CLOSURE) {
#pragma unroll
    for (int i = 0; i < HALF_SPAN; ++i)
      acc += HALF_ROWS[lo][i] * u(i - lo);
  } else {
#pragma unroll
    for (int i = 0; i < HALF_SPAN; ++i)
      acc -= HALF_ROWS[hi][i] * u(hi + 1 - i);
  }
  return acc;
}

// d/dx at node a of a half-point field along a line; past a wall it reads the
// half-grid wall point as zero
template <bool DEEP = false, class L>
__device__ __forceinline__ real_t node_derivative(const L &u, const int a,
                                                  const int N) {
  const int lo = a - 1, hi = N - 2 - a;
  if (!DEEP && (lo < NODE_CLOSURE || hi < NODE_CLOSURE))
    return node_rows(u, lo, hi);
  real_t acc = (real_t)0;
#pragma unroll
  for (int k = 1; k <= STENCIL_RADIUS; ++k)
    acc += SG_W(STENCIL_RADIUS, k) * (u(k - 1) - u(-k));
  return acc;
}

// d/dx at the half point a (a + 1/2) of a nodal field along a line
template <bool DEEP = false, class L>
__device__ __forceinline__ real_t half_derivative(const L &u, const int a,
                                                  const int N) {
  const int lo = a - 1, hi = N - 3 - a;
  if (!DEEP && (lo < HALF_CLOSURE || hi < HALF_CLOSURE))
    return half_rows(u, lo, hi);
  real_t acc = (real_t)0;
#pragma unroll
  for (int k = 1; k <= STENCIL_RADIUS; ++k)
    acc += SG_W(STENCIL_RADIUS, k) * (u(k) - u(1 - k));
  return acc;
}

// the divergence of a nodal stress at the half point a: half_derivative's rows,
// which are minus node_derivative's transpose in the norms, so the norms cancel
// against the inertia's; inside, the taps in the order the deep path rounds
template <bool DEEP = false, class L>
__device__ __forceinline__ real_t node_divergence(const L &sg, const int a,
                                                  const int N) {
  const int lo = a - 1, hi = N - 3 - a;
  if (!DEEP && (lo < HALF_CLOSURE || hi < HALF_CLOSURE))
    return half_rows(sg, lo, hi);
  real_t acc = (real_t)0;
#pragma unroll
  for (int k = 1; k <= STENCIL_RADIUS; ++k) {
    acc += SG_W(STENCIL_RADIUS, k) * sg(k);
    acc -= SG_W(STENCIL_RADIUS, k) * sg(1 - k);
  }
  return acc;
}

// the divergence of a half-point stress at node a: node_derivative's rows, the
// stress at the wall point never read
template <bool DEEP = false, class L>
__device__ __forceinline__ real_t half_divergence(const L &sg, const int a,
                                                  const int N) {
  const int lo = a - 1, hi = N - 2 - a;
  if (!DEEP && (lo < NODE_CLOSURE || hi < NODE_CLOSURE))
    return node_rows(sg, lo, hi);
  real_t acc = (real_t)0;
#pragma unroll
  for (int k = 1; k <= STENCIL_RADIUS; ++k) {
    acc += SG_W(STENCIL_RADIUS, k) * sg(k - 1);
    acc -= SG_W(STENCIL_RADIUS, k) * sg(-k);
  }
  return acc;
}
#endif

// --------------------------------------- pml slabs
#if NDIM == 1
#define PML_PARAMS const int N0
#define PML_GEOM const int n[1] = {N0}, s[1] = {1}
#elif NDIM == 2
#define PML_PARAMS const int N0, const int N1, const int s0
#define PML_GEOM const int n[2] = {N0, N1}, s[2] = {s0, 1}
#elif NDIM == 3
#define PML_PARAMS                                                             \
  const int N0, const int N1, const int s0, const int N2, const int s1
#define PML_GEOM const int n[3] = {N0, N1, N2}, s[3] = {s0, s1, 1}
#endif

// padded extent of axis e: only the fastest is padded, to the stride above it
__device__ __forceinline__ int padded_extent(const int e, const int *n,
                                             const int *s) {
#if NDIM == 1
  return n[0];
#else
  return e < NDIM - 1 ? n[e] : s[NDIM - 2];
#endif
}

// nodes per slab row of axis d: the padded extents of the other axes
__device__ __forceinline__ int slab_plane(const int d, const int *n,
                                          const int *s) {
  int plane = 1;
#pragma unroll
  for (int e = 0; e < NDIM; ++e)
    if (e != d)
      plane *= padded_extent(e, n, s);
  return plane;
}

// slab thread t of axis d, rows lo + hi off its walls, onto an interior node
__device__ __forceinline__ bool slab_node(const int t, const int d,
                                          const int lo, const int hi,
                                          const int *n, const int *s, int &idx,
                                          int &a, int &off, real_t &w) {
  const int plane = slab_plane(d, n, s), rows = lo + hi;
  // the fast axis keeps its threads consecutive, so the slab rows go last there
  const int row = d < NDIM - 1 ? t / plane : t % rows;
  off = d < NDIM - 1 ? t % plane : t / rows;
  if (row >= rows || off >= plane)
    return false;
  a = row < lo ? 1 + row : n[d] - 2 - (row - lo);
  idx = a * s[d];
  w = (a == 1 || a == n[d] - 2) ? (real_t)0.5 : (real_t)1; // cell weight
  int rest = off;
#pragma unroll
  for (int e = NDIM - 1; e >= 0; --e)
    if (e != d) {
      const int p = padded_extent(e, n, s), ae = rest % p;
      rest /= p;
      if (ae < 1 || ae > n[e] - 2)
        return false;
      idx += ae * s[e];
      w *= (ae == 1 || ae == n[e] - 2) ? (real_t)0.5 : (real_t)1;
    }
  return true;
}

// the slab thread of row a beside `off`, or -1 where a lies outside the slab
__device__ __forceinline__ int slab_index(const int a, const int off,
                                          const int d, const int lo,
                                          const int hi, const int *n,
                                          const int plane) {
  const int row = (a >= 1 && a - 1 < lo)                 ? a - 1
                  : (a <= n[d] - 2 && n[d] - 2 - a < hi) ? lo + n[d] - 2 - a
                                                         : -1;
  if (row < 0)
    return -1;
  return d < NDIM - 1 ? row * plane + off : off * (lo + hi) + row;
}

// whether the node at idx is the wall node of a face DIRICHLET_FACES names
__device__ __forceinline__ bool held_node(const int idx, const int *n,
                                          const int *s) {
  bool held = false;
  int rest = idx;
#pragma unroll
  for (int e = 0; e < NDIM; ++e) {
    const int ae = rest / s[e];
    rest -= ae * s[e];
    held |= held_wall(ae, n[e], DIRICHLET_FACES >> 2 * e);
  }
  return held;
}

// ------------------------------- pml memory accessors
// component k of a field j nodes along the axis from a node
struct Field {
  const real_t *__restrict__ p;
  int idx, cs, step;
  __device__ real_t operator()(const int k, const int j) const {
    return p[k * cs + idx + j * step];
  }
};

// a field plus its memory row, through the profile: what the transposes read
template <class Base> struct Memory {
  Base base;
  const real_t *__restrict__ aux, *__restrict__ profile;
  const int *NN;
  int size, row0, a, off, e, lo, hi, plane;
  bool own_half; // whether component e sits on the half point of axis e
  __device__ real_t operator()(const int k, const int j) const {
    const int at = a + j, i = slab_index(at, off, e, lo, hi, NN, plane);
    if (i < 0)
      return (real_t)0; // the profile vanishes off the slab
    const bool half = (k == e) == own_half;
    const real_t coef = profile[(half ? 2 * NN[e] : 0) + at];
    return coef * (base(k, j) + aux[(row0 + k) * size + i]);
  }
};

// a slab node's axes off its flat index, the padded strides S
__device__ __forceinline__ void node_axes(const int idx, const int *S, int *A) {
  int rest = idx;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    A[d] = rest / S[d];
    rest -= A[d] * S[d];
  }
}

#if NDIM == 1
#define PML_AXES                                                               \
  const int NN[1] = {N0}, S[1] = {1};                                          \
  const real_t F[1] = {f0}
#elif NDIM == 2
#define PML_AXES                                                               \
  const int NN[2] = {N0, N1}, S[2] = {s0, 1};                                  \
  const real_t F[2] = {f0, f1}
#elif NDIM == 3
#define PML_AXES                                                               \
  const int NN[3] = {N0, N1, N2}, S[3] = {s0, s1, 1};                          \
  const real_t F[3] = {f0, f1, f2}
#endif

// the slab node of this thread, rows lo + hi plus ext past the front, or return
#define SLAB_OR_RETURN(ext)                                                    \
  int idx, a, off, A[NDIM];                                                    \
  real_t w;                                                                    \
  if (!slab_node(blockIdx.x * blockDim.x + threadIdx.x, e,                     \
                 lo ? lo + (ext) : 0, hi ? hi + (ext) : 0, NN, S, idx, a, off, \
                 w))                                                           \
    return;                                                                    \
  node_axes(idx, S, A);                                                        \
  const int plane = slab_plane(e, NN, S), size = (lo + hi) * plane;            \
  const int t = slab_index(a, off, e, lo, hi, NN, plane)

// ---------------------------------- transfer kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
excitation_kernel(real_t *__restrict__ u, const real_t *__restrict__ source,
                  const int offset, const int *__restrict__ lin_index,
                  const int num_sources, const real_t *__restrict__ weight) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sources) {
    atomicAdd(&u[lin_index[idx]], weight[idx] * source[offset + idx]);
  }
}

// ------------------------------------------------------------------------------------
__global__ void adjoint_excitation_kernel(
    real_t *__restrict__ l2, const real_t *__restrict__ signal,
    const int offset, const int *__restrict__ lin_index, const int num_sensors,
    const real_t *__restrict__ weight, real_t *__restrict__ g_mass,
    const real_t *__restrict__ u1, const real_t mf) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sensors) {
    const int n = lin_index[idx];
    const real_t load = weight[idx] * signal[offset + idx];
    atomicAdd(&l2[n], load);
    // the injected load is part of the second time difference of lambda
    atomicAdd(&g_mass[n], -mf * u1[n] * load);
  }
}

// ------------------------------------------------------------------------------------
__global__ void get_signal_kernel(const real_t *__restrict__ u,
                                  real_t *__restrict__ um, const int offset,
                                  const int *__restrict__ lin_index,
                                  const int num_sensors) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sensors) {
    um[offset + idx] = u[lin_index[idx]];
  }
}

// ------------------------------------------------------------------------------------
__global__ void set_signal_kernel(real_t *__restrict__ u,
                                  const real_t *__restrict__ um,
                                  const int offset,
                                  const int *__restrict__ lin_index,
                                  const int num_sensors) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sensors) {
    u[lin_index[idx]] = um[offset + idx];
  }
}

// ------------------------------------------------------------------------------------
__global__ void dft_kernel(const real_t *__restrict__ u,
                           real_t *__restrict__ acc,
                           const real_t *__restrict__ table, const int offset,
                           const int *__restrict__ lin_index, const int num,
                           const int num_rows) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    const real_t uc = u[lin_index[idx]];
    for (int k = 0; k < num_rows; ++k) {
      acc[k * num + idx] += table[offset + k] * uc; // cos, -sin per frequency
    }
  }
}

} // extern "C"
