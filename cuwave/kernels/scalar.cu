// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3
//   USE_DAMPING
//   USE_DOMAIN

// spatial finite difference stencil for Laplacian
__device__ __forceinline__ real_t flux_divergence_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ stiff,
    const int idx, const int s, const real_t uc, const real_t sc,
    const real_t factor, const int r) {
  const real_t sp = stiff[idx + s];            // plus of sc
  const real_t sm = stiff[idx - s];            // minus of sc
  const real_t gp = sc * sp / (sc + sp);       // harmonic mean
  const real_t gm = sc * sm / (sc + sm);       // harmonic mean
  real_t Dp = OP_W(r, 1) * (u1[idx + s] - uc); // initialization: inner grad
  real_t Dm = OP_W(r, 1) * (uc - u1[idx - s]); // initialization: inner grad
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Dp +=
          OP_W(r, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]); // inner grad
      Dm +=
          OP_W(r, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]); // inner grad
    }
  return factor * (Dp * gp - Dm * gm); // outer grad (incl. inner grad)
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


} // extern "C"
