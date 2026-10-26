// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3
//   USE_DAMPING
//   USE_DOMAIN
//   USE_STREAM, STREAM_TX
//   STEPS, TILE0, TILE1, ROWS (2D)

// spatial finite difference stencil for Laplacian, `u(j)` the tap j nodes on
template <class Taps>
__device__ __forceinline__ real_t flux_divergence_taps(
    const Taps &u, const real_t uc, const real_t gp, const real_t gm,
    const real_t factor, const int r) {
  // rounded as nvcc contracts the plain sums, so every caller agrees bitwise
  real_t Dp = u(1) - uc, Dm = uc - u(-1); // inner grad
  if (STENCIL_RADIUS > 1) { // a weight of one is exact, and an intrinsic stays
    Dp = MUL_RN(OP_W(r, 1), Dp);
    Dm = MUL_RN(OP_W(r, 1), Dm);
  }
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Dp = FMA_RN(OP_W(r, k), u(k) - u(1 - k), Dp);
      Dm = FMA_RN(OP_W(r, k), u(k - 1) - u(-k), Dm);
    }
  return factor * FMA_RN(Dp, gp, -MUL_RN(Dm, gm)); // outer grad
}

// the same with the taps strided `s` from `idx` in `u1`
__device__ __forceinline__ real_t flux_divergence_means(
    const real_t *__restrict__ u1, const int idx, const int s, const real_t uc,
    const real_t gp, const real_t gm, const real_t factor, const int r) {
  const auto u = [&](const int j) { return u1[idx + j * s]; };
  return flux_divergence_taps(u, uc, gp, gm, factor, r);
}

// spatial finite difference stencil for Laplacian
__device__ __forceinline__ real_t flux_divergence_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ stiff,
    const int idx, const int s, const real_t uc, const real_t sc,
    const real_t factor, const int r) {
  const real_t sp = stiff[idx + s]; // plus of sc
  const real_t sm = stiff[idx - s]; // minus of sc
  return flux_divergence_means(u1, idx, s, uc, sc * sp / (sc + sp),
                               sc * sm / (sc + sm), factor, r);
}

// ------------------------------------- pml helpers
// the two cell differences of flux_divergence_axis, before the stiffness
__device__ __forceinline__ void cell_differences(const real_t *__restrict__ u1,
                                                 const int idx, const int s,
                                                 const real_t uc, const int r,
                                                 real_t &Dp, real_t &Dm) {
  Dp = OP_W(r, 1) * (u1[idx + s] - uc);
  Dm = OP_W(r, 1) * (uc - u1[idx - s]);
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Dp += OP_W(r, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]);
      Dm += OP_W(r, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]);
    }
}

// ----------------------------------- domain helpers
#ifdef USE_DOMAIN
#define DIRICHLET 7 // the radius code of a cell open onto a node held at zero

// the same with each cell at its own radius, zero for a cell leaving the domain
__device__ __forceinline__ real_t flux_divergence_cells(
    const real_t *__restrict__ u1, const real_t *__restrict__ stiff,
    const int idx, const int s, const real_t uc, const real_t sc,
    const real_t factor, const int rp, const int rm) {
  real_t Dp = 0, Dm = 0; // a closed cell carries no flux
  if (rp == DIRICHLET)
    Dp = -uc * sc; // onto a zero halfway across, at the node's own stiffness
  else if (rp) {
    const real_t sp = stiff[idx + s];
    Dp = OP_W(rp, 1) * (u1[idx + s] - uc);
#pragma unroll
    for (int k = 2; k <= STENCIL_RADIUS; ++k)
      if (k <= rp)
        Dp += OP_W(rp, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]);
    Dp *= sc * sp / (sc + sp); // harmonic mean
  }
  if (rm == DIRICHLET)
    Dm = uc * sc;
  else if (rm) {
    const real_t sm = stiff[idx - s];
    Dm = OP_W(rm, 1) * (uc - u1[idx - s]);
#pragma unroll
    for (int k = 2; k <= STENCIL_RADIUS; ++k)
      if (k <= rm)
        Dm += OP_W(rm, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]);
    Dm *= sc * sm / (sc + sm); // harmonic mean
  }
  return factor * (Dp - Dm);
}
#endif

#ifdef USE_DOMAIN
#define BLOCK_X (tile & 1023) // one block per domain tile, 10 bits an axis
#define BLOCK_Y ((tile >> 10) & 1023)
#define BLOCK_Z ((tile >> 20) & 1023)
#else
#define BLOCK_X blockIdx.x
#define BLOCK_Y blockIdx.y
#define BLOCK_Z blockIdx.z
#endif

#ifdef USE_DOMAIN
#define CELLS(s, f, d)                                                         \
  flux_divergence_cells(u1, stiff, idx, s, uc, sc, f, (code >> 6 * (d)) & 7,   \
                        (code >> (6 * (d) + 3)) & 7)

// out of line, so the rare wall node costs the deep ones no registers
__device__ __noinline__ real_t
wall_laplacian(const real_t *__restrict__ u1, const real_t *__restrict__ stiff,
               const int idx, const real_t uc, const real_t sc, const int code,
               const real_t f0
#if NDIM >= 2
               ,
               const real_t f1, const int s0
#endif
#if NDIM >= 3
               ,
               const real_t f2, const int s1
#endif
) {
#if NDIM == 1
  return CELLS(1, f0, 0);
#elif NDIM == 2
  return CELLS(s0, f0, 0) + CELLS(1, f1, 1);
#elif NDIM == 3
  return CELLS(s0, f0, 0) + CELLS(s1, f1, 1) + CELLS(1, f2, 2);
#endif
}
#endif

// ----------------------------- boundary condition helper
#if NDIM == 1
#define BC_PARAMS const int N0
#define BC_GEOM const int n[1] = {N0}, s[1] = {1}
#elif NDIM == 2
#define BC_PARAMS const int N0, const int N1, const int s0
#define BC_GEOM const int n[2] = {N0, N1}, s[2] = {s0, 1}
#elif NDIM == 3
#define BC_PARAMS                                                              \
  const int N0, const int N1, const int s0, const int N2, const int s1
#define BC_GEOM const int n[3] = {N0, N1, N2}, s[3] = {s0, s1, 1}
#endif

__device__ __forceinline__ bool bc_ghost(int t, const int faces, const int *n,
                                         const int *s, int &ghost,
                                         int &normal) {
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    int face = 1; // nodes on one ghost face of axis d
#pragma unroll
    for (int k = 0; k < NDIM; ++k)
      if (k != d)
        face *= n[k] - 2;

    const int lo = (faces >> (2 * d)) & 1, hi = (faces >> (2 * d + 1)) & 1;
    if (t < (lo + hi) * face) {
      const int high =
          (lo && t < face) ? 0 : 1; // low ghost (0) or high (n[d] - 1)
      int r = t - (high ? lo * face : 0);
      int off = 0; // position within the face, axis d excluded
#pragma unroll
      for (int k = NDIM - 1; k >= 0; --k)
        if (k != d) {
          off += (r % (n[k] - 2) + 1) * s[k];
          r /= n[k] - 2;
        }
      ghost = off + (high ? n[d] - 1 : 0) * s[d];
      normal = high ? -s[d] : s[d];
      return true;
    }
    t -= (lo + hi) * face;
  }
  return false;
}

// -------------------------------------- kernels
extern "C" {

#ifdef USE_STREAM
// ------------------------------------------------------------------------------------
__global__ void
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ stiff,
          const real_t *__restrict__ minv, const int derive_inertia,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
          const int chunk, const real_t f0, const int N0, const real_t f1,
          const int N1, const int s0) {
  // u and stiff of the current row, tiled with a halo of R both alike
  __shared__ real_t su[STREAM_TX + 2 * STENCIL_RADIUS];
  __shared__ real_t sk[STREAM_TX + 2 * STENCIL_RADIUS];
  const int tx = threadIdx.x, x0 = blockIdx.x * STREAM_TX, a1 = x0 + tx;
  const int cx = tx + STENCIL_RADIUS; // this node in the tile
  const bool in_row = a1 < N1, inner = a1 > 0 && a1 < N1 - 1;
  const int r1 = CLOSURE(a1, N1);
  const int a_first = 1 + blockIdx.y * chunk;
  const int a_last = min(N0 - 1, a_first + chunk);

  // axis 0 marches, its taps and three stiffnesses queued in registers
  real_t q[2 * STENCIL_RADIUS + 1]; // u on rows a0 - R .. a0 + R
#pragma unroll
  for (int k = 1; k <= 2 * STENCIL_RADIUS; ++k) {
    const int row = a_first - STENCIL_RADIUS - 1 + k;
    q[k] = (in_row && row >= 0 && row < N0) ? u1[row * s0 + a1] : (real_t)0;
  }
  real_t sm, sc = in_row ? stiff[(a_first - 1) * s0 + a1] : (real_t)1;
  real_t sp = in_row ? stiff[a_first * s0 + a1] : (real_t)1;
  for (int a0 = a_first; a0 < a_last; ++a0) {
    const int idx = a0 * s0 + a1;
#pragma unroll
    for (int k = 0; k < 2 * STENCIL_RADIUS; ++k)
      q[k] = q[k + 1];
    const int ahead = a0 + STENCIL_RADIUS;
    q[2 * STENCIL_RADIUS] =
        (in_row && ahead < N0) ? u1[ahead * s0 + a1] : (real_t)0;
    sm = sc;
    sc = sp;
    sp = in_row ? stiff[idx + s0] : (real_t)1; // row a0 + 1 is at most a ghost
    const real_t old = inner ? u0[idx] : (real_t)0;
    __syncthreads(); // the previous row's tile is consumed

    su[cx] = q[STENCIL_RADIUS];
    sk[cx] = sc;
    for (int e = tx; e < STENCIL_RADIUS; e += STREAM_TX) {
      const int xl = x0 - STENCIL_RADIUS + e, xr = x0 + STREAM_TX + e;
      su[e] = xl >= 0 ? u1[a0 * s0 + xl] : (real_t)0;
      su[STENCIL_RADIUS + STREAM_TX + e] =
          xr < N1 ? u1[a0 * s0 + xr] : (real_t)0;
    }
    if (tx == 0) // the cell stiffness reaches one node either way
      sk[cx - 1] = x0 > 0 ? stiff[idx - 1] : (real_t)1;
    if (tx == STREAM_TX - 1)
      sk[cx + 1] = a1 + 1 < N1 ? stiff[idx + 1] : (real_t)1;
    __syncthreads();
    if (!inner)
      continue;

    // axis 0 from the queue, in the order flux_divergence_axis takes its taps
    const real_t uc = q[STENCIL_RADIUS];
    const int r0 = CLOSURE(a0, N0);
    const real_t gp = sc * sp / (sc + sp), gm = sc * sm / (sc + sm);
    real_t Dp = OP_W(r0, 1) * (q[STENCIL_RADIUS + 1] - uc);
    real_t Dm = OP_W(r0, 1) * (uc - q[STENCIL_RADIUS - 1]);
#pragma unroll
    for (int k = 2; k <= STENCIL_RADIUS; ++k)
      if (k <= r0) {
        Dp += OP_W(r0, k) * (q[STENCIL_RADIUS + k] - q[STENCIL_RADIUS - k + 1]);
        Dm += OP_W(r0, k) * (q[STENCIL_RADIUS + k - 1] - q[STENCIL_RADIUS - k]);
      }
    const real_t laplacian =
        f0 * (Dp * gp - Dm * gm) +
        flux_divergence_axis(su + cx, sk + cx, 0, 1, uc, sc, f1, r1);

    const real_t mi = derive_inertia ? 1.f / sc : minv[idx];
#ifdef USE_DAMPING
    const real_t beta = 0.5f * mi * damping[idx] * dt;
    u2[idx] = (2.f * uc - old * (1.f - beta) + mi * laplacian) / (1.f + beta);
#else
    u2[idx] = -old + 2.f * uc + mi * laplacian;
#endif
  }
}
#else
// ------------------------------------------------------------------------------------
__global__ void
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ stiff,
          const real_t *__restrict__ minv, const int derive_inertia,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
#ifdef USE_DOMAIN
          const int *__restrict__ cells, const int *__restrict__ tiles,
#endif
          const real_t f0, const int N0
#if NDIM >= 2
          ,
          const real_t f1, const int N1, const int s0
#endif
#if NDIM >= 3
          ,
          const real_t f2, const int N2, const int s1
#endif
) {
#ifdef USE_DOMAIN
  const int tile = tiles[blockIdx.x];
#endif
#if NDIM == 1
  const int a0 = BLOCK_X * blockDim.x + threadIdx.x;
  if (!(a0 > 0 && a0 < N0 - 1))
    return;
  const int idx = a0;
  const int r0 = CLOSURE(a0, N0);
#elif NDIM == 2
  const int a1 = BLOCK_X * blockDim.x + threadIdx.x;
  const int a0 = BLOCK_Y * blockDim.y + threadIdx.y;
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1))
    return;
  const int idx = a0 * s0 + a1;
  const int r0 = CLOSURE(a0, N0), r1 = CLOSURE(a1, N1);
#elif NDIM == 3
  const int a2 = BLOCK_X * blockDim.x + threadIdx.x;
  const int a1 = BLOCK_Y * blockDim.y + threadIdx.y;
  const int a0 = BLOCK_Z * blockDim.z + threadIdx.z;
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&
        a2 < N2 - 1))
    return;
  const int idx = a0 * s0 + a1 * s1 + a2;
  const int r0 = CLOSURE(a0, N0), r1 = CLOSURE(a1, N1), r2 = CLOSURE(a2, N2);
#endif

  const real_t uc = u1[idx];    // load once
  const real_t sc = stiff[idx]; // load once
#if NDIM == 1
  real_t laplacian = flux_divergence_axis(u1, stiff, idx, 1, uc, sc, f0, r0);
#elif NDIM == 2
  real_t laplacian = flux_divergence_axis(u1, stiff, idx, s0, uc, sc, f0, r0) +
                     flux_divergence_axis(u1, stiff, idx, 1, uc, sc, f1, r1);
#elif NDIM == 3
  real_t laplacian = flux_divergence_axis(u1, stiff, idx, s0, uc, sc, f0, r0) +
                     flux_divergence_axis(u1, stiff, idx, s1, uc, sc, f1, r1) +
                     flux_divergence_axis(u1, stiff, idx, 1, uc, sc, f2, r2);
#endif
#ifdef USE_DOMAIN
  // bit 30 marks a tile touching the wall, so the branch is uniform per block
  if (tile >> 30) {
    const int code = cells[idx]; // 0 outside, bit 30 deep, else the cell radii
    if (!code)
      return; // outside the domain nothing is stepped
#if NDIM == 1
    if (!(code >> 30))
      laplacian = wall_laplacian(u1, stiff, idx, uc, sc, code, f0);
#elif NDIM == 2
    if (!(code >> 30))
      laplacian = wall_laplacian(u1, stiff, idx, uc, sc, code, f0, f1, s0);
#elif NDIM == 3
    if (!(code >> 30))
      laplacian =
          wall_laplacian(u1, stiff, idx, uc, sc, code, f0, f1, s0, f2, s1);
#endif
  }
#endif

  const real_t mi = derive_inertia ? 1.f / sc : minv[idx];
#ifdef USE_DAMPING
  const real_t beta = 0.5f * mi * damping[idx] * dt;
  u2[idx] = (2.f * uc - u0[idx] * (1.f - beta) + mi * laplacian) / (1.f + beta);
#else
  u2[idx] = -u0[idx] + 2.f * uc + mi * laplacian;
#endif
}
#endif

// ------------------------------------------------------------------------------------
__global__ void
pml_kernel(const real_t *__restrict__ u1, real_t *__restrict__ u2,
           const real_t *__restrict__ stiff, const real_t *__restrict__ minv,
           const int derive_inertia, real_t *__restrict__ aux,
           const real_t *__restrict__ profile, const int d, const int lo,
           const int hi, const real_t f, PML_PARAMS) {
  PML_GEOM;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  int idx, a, off;
  real_t w;
  if (!slab_node(t, d, lo, hi, n, s, idx, a, off, w))
    return;
  const int N = n[d], o = s[d], size = (lo + hi) * slab_plane(d, n, s);
  const real_t *an = profile, *bn = profile + N;             // at the nodes
  const real_t *ah = profile + 2 * N, *bh = profile + 3 * N; // at a + 1/2

  const real_t uc = u1[idx], sc = stiff[idx];
  real_t Dp, Dm;
  cell_differences(u1, idx, o, uc, CLOSURE(a, N), Dp, Dm);
  const real_t sp = stiff[idx + o], sm = stiff[idx - o];
  const real_t gp = sc * sp / (sc + sp), gm = sc * sm / (sc + sm); // harmonic

  const real_t pp = bh[a] * aux[t] + ah[a] * Dp;                // plus cell
  const real_t pm = bh[a - 1] * aux[size + t] + ah[a - 1] * Dm; // minus cell
  const real_t z = bn[a] * aux[2 * size + t] +
                   an[a] * f * (gp * (Dp + pp) - gm * (Dm + pm)); // divergence
  aux[t] = pp;
  aux[size + t] = pm;
  aux[2 * size + t] = z;
  const real_t mi = derive_inertia ? 1.f / sc : minv[idx];
  u2[idx] += mi * (f * (gp * pp - gm * pm) + z);
}

// ------------------------------------------------------------------------------------
__global__ void homogeneous_neumann_kernel(real_t *__restrict__ u,
                                           const int faces, BC_PARAMS) {
  BC_GEOM;
  int ghost, normal;
  if (!bc_ghost(blockIdx.x * blockDim.x + threadIdx.x, faces, n, s, ghost,
                normal))
    return;
  u[ghost] = u[ghost + 2 * normal];
}

// ------------------------------------------------------------------------------------
__global__ void homogeneous_dirichlet_kernel(real_t *__restrict__ u,
                                             const int faces, BC_PARAMS) {
  BC_GEOM;
  int ghost, normal;
  if (!bc_ghost(blockIdx.x * blockDim.x + threadIdx.x, faces, n, s, ghost,
                normal))
    return;
  u[ghost] = -u[ghost + 2 * normal];
}

#ifdef STEPS
// ------------------------------------------------------------------------------------
// the core a tile writes back, its halo STEPS stencil radii wide
#define HALO (STEPS * STENCIL_RADIUS)
#define C0 (TILE0 - 2 * HALO)
#define C1 (TILE1 - 2 * HALO)
#define NODES (TILE0 / ROWS) // a thread's consecutive nodes along axis 0
#define WIDE (TILE1 + 2 * STENCIL_RADIUS) // a tile row with its guard columns
#define AREA ((TILE0 + 2 * STENCIL_RADIUS) * WIDE)
#define AT(x, y) (((x) + STENCIL_RADIUS) * WIDE + (y) + STENCIL_RADIUS)

// advance a tile STEPS steps between one load and one store: the step, the
// excitation, the boundary refill and the record, bitwise the plain launches
__global__ void __launch_bounds__(TILE1 *ROWS) blocked_kernel(
    const real_t *__restrict__ u0, const real_t *__restrict__ u1,
    real_t *__restrict__ v0, real_t *__restrict__ v1,
    const real_t *__restrict__ stiff, const real_t *__restrict__ minv,
    const int derive_inertia,
#ifdef USE_DAMPING
    const real_t *__restrict__ damping, const real_t dt,
#endif
    const real_t *__restrict__ signal, const real_t *__restrict__ weights,
    const int num_sources, real_t *__restrict__ record, const int num_sensors,
    const int *__restrict__ source_starts,
    const unsigned *__restrict__ source_entries,
    const int *__restrict__ sensor_starts,
    const unsigned *__restrict__ sensor_entries, const int t0, const int neumann,
    const int dirichlet, const real_t f0, const int N0, const real_t f1,
    const int N1, const int s0) {
  // u of this step and the next, each in a guard ring a radius wide
  extern __shared__ real_t shared[];
  const int ty = threadIdx.x, x0 = threadIdx.y * NODES;
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  const int lo0 = 1 + blockIdx.y * C0, lo1 = 1 + blockIdx.x * C1;
  const int a1 = lo1 - HALO + ty, k1 = min(max(a1, 1), N1 - 2);
  const bool col = a1 >= 0 && a1 <= N1 - 1;
  const bool core1 = k1 >= lo1 && k1 < lo1 + C1;

  // the thread's nodes, loaded once
  real_t up[NODES], uc[NODES], mi[NODES], sc[NODES];
#ifdef USE_DAMPING
  real_t beta[NODES];
#endif
  unsigned stepped = 0, ghost = 0;
#pragma unroll
  for (int k = 0; k < NODES; ++k) {
    const int x = x0 + k, a0 = lo0 - HALO + x;
    const bool inside = col && a0 >= 0 && a0 <= N0 - 1;
    const bool interior = a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1;
    const int idx = a0 * s0 + a1;
    up[k] = inside ? u0[idx] : 0;
    uc[k] = inside ? u1[idx] : 0;
    sc[k] = inside ? stiff[idx] : (real_t)1;
    shared[AT(x, ty)] = uc[k];
    shared[AREA + AT(x, ty)] = sc[k];
    mi[k] = interior ? (derive_inertia ? 1.f / stiff[idx] : minv[idx]) : 0;
#ifdef USE_DAMPING
    beta[k] = interior ? 0.5f * mi[k] * damping[idx] : 0; // over dt
#endif
    stepped |= (unsigned)interior << k;
    // a ghost of a face the refill reaches, edges and corners excluded
    const bool along0 = a1 > 0 && a1 < N1 - 1, along1 = a0 > 0 && a0 < N0 - 1;
    const int f = along0 && a0 == 0        ? 0
                  : along0 && a0 == N0 - 1 ? 1
                  : along1 && a1 == 0      ? 2
                  : along1 && a1 == N1 - 1 ? 3
                                           : -1;
    ghost |= (unsigned)(f >= 0 && (((neumann | dirichlet) >> f) & 1)) << k;
  }
  const bool refill = __syncthreads_or(ghost);

  // the harmonic cell stiffnesses: g0[k] between nodes k - 1 and k of the
  // thread's run, gm1 and gp1 toward -1 and +1, from the staged stiffness
  const real_t *sk = shared + AREA;
  real_t g0[NODES + 1], gm1[NODES], gp1[NODES];
#pragma unroll
  for (int k = 0; k <= NODES; ++k) {
    const real_t a = k > 0 ? sc[k - 1] : sk[AT(x0 - 1, ty)];
    const real_t b = k < NODES ? sc[k] : sk[AT(x0 + NODES, ty)];
    g0[k] = a * b / (a + b);
  }
#pragma unroll
  for (int k = 0; k < NODES; ++k) {
    const real_t m = sk[AT(x0 + k, ty - 1)], p = sk[AT(x0 + k, ty + 1)];
    gm1[k] = m * sc[k] / (m + sc[k]);
    gp1[k] = sc[k] * p / (sc[k] + p);
  }
  __syncthreads();

  // the tile's sources and sensors as tags, source + 1 and sensor + 1 a half
  unsigned *tags = (unsigned *)(shared + AREA);
#pragma unroll
  for (int k = 0; k < NODES; ++k)
    tags[AT(x0 + k, ty)] = 0;
  __syncthreads();
  const int thread = threadIdx.y * TILE1 + ty;
  for (int e = source_starts[tile] + thread; e < source_starts[tile + 1];
       e += TILE1 * ROWS) {
    const unsigned entry = source_entries[e], local = entry & 0xffff;
    atomicOr(&tags[AT(local / TILE1, local % TILE1)], (entry >> 16) + 1);
  }
  for (int e = sensor_starts[tile] + thread; e < sensor_starts[tile + 1];
       e += TILE1 * ROWS) {
    const unsigned entry = sensor_entries[e], local = entry & 0xffff;
    atomicOr(&tags[AT(local / TILE1, local % TILE1)], (entry >> 16) + 1 << 16);
  }
  __syncthreads();
  unsigned tag[NODES];
#pragma unroll
  for (int k = 0; k < NODES; ++k)
    tag[k] = tags[AT(x0 + k, ty)];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < NODES; ++k)
    shared[AREA + AT(x0 + k, ty)] = uc[k];
  __syncthreads();

  for (int t = 0; t < STEPS; ++t) {
    // the tile edge reads garbage, which never reaches the core
    const real_t *cu = shared + (t & 1) * AREA;
    real_t *nu = shared + (~t & 1) * AREA;
#pragma unroll
    for (int k = 0; k < NODES; ++k) {
      if (!((stepped >> k) & 1))
        continue;
      const int x = x0 + k, a0 = lo0 - HALO + x, i = AT(x, ty);
      // axis 0 from the thread's own run, past its ends from the tile
      const auto along0 = [&](const int j) {
        return k + j >= 0 && k + j < NODES ? uc[k + j] : cu[i + j * WIDE];
      };
      const auto along1 = [&](const int j) { return cu[i + j]; };
      const real_t d0 = flux_divergence_taps(along0, uc[k], g0[k + 1], g0[k],
                                             1, CLOSURE(a0, N0));
      const real_t d1 = flux_divergence_taps(along1, uc[k], gp1[k], gm1[k], 1,
                                             CLOSURE(a1, N1));
      // one axis fused onto the other, as nvcc contracts fd_kernel's sum
#if STENCIL_RADIUS == 1
      const real_t laplacian = FMA_RN(f0, d0, MUL_RN(f1, d1));
#else
      const real_t laplacian = FMA_RN(f1, d1, MUL_RN(f0, d0));
#endif
      // the leapfrog rounded as nvcc compiles fd_kernel's
#ifdef USE_DAMPING
      real_t u2 =
          FMA_RN(mi[k], laplacian,
                 FMA_RN(-up[k], FMA_RN(-dt, beta[k], 1.f), 2.f * uc[k])) /
          FMA_RN(dt, beta[k], 1.f);
#else
      real_t u2 = FMA_RN(mi[k], laplacian, 2.f * uc[k] - up[k]);
#endif
      // the product rounded apart, as excitation_kernel adds it atomically
      const int j = (tag[k] & 0xffffu) - 1;
      if (j >= 0)
        u2 += MUL_RN(weights[j], signal[(t0 + t) * num_sources + j]);
      up[k] = u2; // u^(t + 1) in the slot of u^(t - 1), swapped below
      nu[i] = u2;
    }
#pragma unroll
    for (int k = 0; k < NODES; ++k)
      if ((stepped >> k) & 1) {
        const real_t next = up[k];
        up[k] = uc[k];
        uc[k] = next;
      }

    // the boundary refill: a ghost takes its mirror two nodes in, the sign held
    if (refill) {
      __syncthreads();
#pragma unroll
      for (int k = 0; k < NODES; ++k)
        if ((ghost >> k) & 1) {
          const int x = x0 + k, a0 = lo0 - HALO + x;
          const int f = a0 == 0 ? 0 : a0 == N0 - 1 ? 1 : a1 == 0 ? 2 : 3;
          const real_t m = nu[AT(x + 2 * (f == 0) - 2 * (f == 1),
                                 ty + 2 * (f == 2) - 2 * (f == 3))];
          up[k] = uc[k];
          uc[k] = (dirichlet >> f) & 1 ? -m : m;
          nu[AT(x, ty)] = uc[k];
        }
    }
#pragma unroll
    for (int k = 0; k < NODES; ++k)
      if (tag[k] >> 16)
        record[(t0 + t) * num_sensors + (tag[k] >> 16) - 1] = uc[k];
    __syncthreads();
  }

  // write back the core, each ghost with the interior node it clamps to
  if (!col || !core1)
    return;
#pragma unroll
  for (int k = 0; k < NODES; ++k) {
    const int a0 = lo0 - HALO + x0 + k;
    const int k0 = min(max(a0, 1), N0 - 2);
    if (a0 >= 0 && a0 <= N0 - 1 && k0 >= lo0 && k0 < lo0 + C0) {
      v0[a0 * s0 + a1] = up[k];
      v1[a0 * s0 + a1] = uc[k];
    }
  }
}
#endif

} // extern "C"
