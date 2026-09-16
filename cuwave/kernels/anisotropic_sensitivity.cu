// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3

#ifndef RADIUS
#define RADIUS 1 // default order 2
#endif

#define BLK (2 * RADIUS)
#if NDIM == 1
#define CELLS BLK
#elif NDIM == 2
#define CELLS (BLK * BLK)
#else
#define CELLS (BLK * BLK * BLK)
#endif
#define NLOC (CELLS * NDIM)
#define CORNERS (1 << NDIM)

// position along axis d of the block entry m, axis 0 running fastest
__device__ __forceinline__ int block_axis(int m, const int d) {
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
    if (k < d)
      m /= BLK;
  return m % BLK;
}

// ----------------------------- cell assembly helpers
#if NDIM == 1
#define AXIS_PARAMS const int N0
#define INTERIOR_OR_RETURN                                                     \
  const int a0 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  if (!(a0 > 0 && a0 < N0 - 1))                                                \
    return;                                                                    \
  const int idx = a0
#define AXIS_GEOM const int A[1] = {a0}, S[1] = {1}, NN[1] = {N0}
#elif NDIM == 2
#define AXIS_PARAMS const int N0, const int N1, const int s0
#define INTERIOR_OR_RETURN                                                     \
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;                        \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1))                       \
    return;                                                                    \
  const int idx = a0 * s0 + a1
#define AXIS_GEOM                                                              \
  const int A[2] = {a0, a1}, S[2] = {s0, 1}, NN[2] = {N0, N1}
#elif NDIM == 3
#define AXIS_PARAMS                                                            \
  const int N0, const int N1, const int s0, const int N2, const int s1
#define INTERIOR_OR_RETURN                                                     \
  const int a2 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  const int a1 = blockIdx.y * blockDim.y + threadIdx.y;                        \
  const int a0 = blockIdx.z * blockDim.z + threadIdx.z;                        \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&            \
        a2 < N2 - 1))                                                          \
    return;                                                                    \
  const int idx = a0 * s0 + a1 * s1 + a2
#define AXIS_GEOM                                                              \
  const int A[3] = {a0, a1, a2}, S[3] = {s0, s1, 1}, NN[3] = {N0, N1, N2}
#endif

// the cell this node is the low corner of, where its density accumulates, plus
// the radius that cell carries once graded down towards a wall
__device__ __forceinline__ bool own_cell(const int *A, const int *NN,
                                         int &radius) {
  radius = RADIUS;
  bool inside = true;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    inside &= (A[d] >= 1 && A[d] <= NN[d] - 3);
    radius = min(radius, min(RADIUS, min(A[d], NN[d] - 2 - A[d])));
  }
  return inside;
}

// block entry l of that cell, and whether it falls inside the graded support
__device__ __forceinline__ bool own_offset(const int l, const int radius,
                                           const int *S, int &off) {
  off = 0;
  bool carries = true;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int m = block_axis(l, d);
    carries &= (m >= RADIUS - radius && m <= RADIUS + radius - 1);
    off += (m - RADIUS + 1) * S[d];
  }
  return carries;
}

// the 2**NDIM cells a node is a corner of: the material stays cell local,
// so the design chain rule runs over those, not the wide stencil block
__device__ __forceinline__ bool corner_cell(const int c, const int *A,
                                            const int *NN, const int *S,
                                            int &base) {
  base = 0;
  bool inside = true;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int bit = (c >> d) & 1;
    inside &= bit ? (A[d] < NN[d] - 2) : (A[d] > 1);
    base += (bit - 1) * S[d];
  }
  return inside;
}

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
gradient_kernel(real_t *__restrict__ g_mass, real_t *__restrict__ g_cell,
                const real_t *__restrict__ u0, const real_t *__restrict__ u1,
                const real_t *__restrict__ u2, const real_t *__restrict__ l1,
                const real_t *__restrict__ stencil, const real_t mass_factor,
                const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;

  // dJ/dmass: no neighbour and no material load
  real_t acc = (real_t)0;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    const int n = i * cs + idx;
    acc += l1[n] * (u2[n] - 2.f * u1[n] + u0[n]);
  }
  g_mass[idx] -= mass_factor * acc;

  // dJ/dcell: the cell stencil paired between the forward and adjoint fields
  int radius;
  if (!own_cell(A, NN, radius))
    return;
  const real_t *__restrict__ table = stencil + (radius - 1) * NLOC * NLOC;
  real_t bilinear = (real_t)0;
#pragma unroll
  for (int l = 0; l < CELLS; ++l) {
    int lo;
    if (!own_offset(l, radius, S, lo))
      continue;
#pragma unroll
    for (int m = 0; m < CELLS; ++m) {
      int mo;
      if (!own_offset(m, radius, S, mo))
        continue;
#pragma unroll
      for (int i = 0; i < NDIM; ++i)
#pragma unroll
        for (int j = 0; j < NDIM; ++j)
          bilinear += l1[i * cs + idx + lo] *
                      table[(l * NDIM + i) * NLOC + m * NDIM + j] *
                      u1[j * cs + idx + mo];
    }
  }
  g_cell[idx] -= bilinear;
}

// ------------------------------------------------------------------------------------
__global__ void
frechet_kernel(real_t *__restrict__ acc_mass, real_t *__restrict__ acc_cell,
               const real_t *__restrict__ u0, const real_t *__restrict__ u1,
               const real_t *__restrict__ u2,
               const real_t *__restrict__ stencil, const real_t ft,
               const real_t fs, const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;

  real_t acc = (real_t)0;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    const int n = i * cs + idx;
    const real_t dudt = u2[n] - u0[n];
    acc += dudt * dudt;
  }
  acc_mass[idx] += ft * acc;

  int radius;
  if (!own_cell(A, NN, radius))
    return;
  const real_t *__restrict__ table = stencil + (radius - 1) * NLOC * NLOC;
  real_t bilinear = (real_t)0;
#pragma unroll
  for (int l = 0; l < CELLS; ++l) {
    int lo;
    if (!own_offset(l, radius, S, lo))
      continue;
#pragma unroll
    for (int m = 0; m < CELLS; ++m) {
      int mo;
      if (!own_offset(m, radius, S, mo))
        continue;
#pragma unroll
      for (int i = 0; i < NDIM; ++i)
#pragma unroll
        for (int j = 0; j < NDIM; ++j)
          bilinear += u1[i * cs + idx + lo] *
                      table[(l * NDIM + i) * NLOC + m * NDIM + j] *
                      u1[j * cs + idx + mo];
    }
  }
  acc_cell[idx] += fs * bilinear;
}

// ------------------------------------------------------------------------------------
__global__ void cell_to_node_kernel(real_t *__restrict__ g_stiff,
                                    const real_t *__restrict__ g_cell,
                                    const real_t *__restrict__ cell,
                                    const real_t *__restrict__ gamma,
                                    const real_t share, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;

  // d(harmonic cell mean)/d(corner) is (cell / corner)^2 over the corner count
  const real_t gn = gamma[idx];
  real_t acc = (real_t)0;
#pragma unroll
  for (int c = 0; c < CORNERS; ++c) {
    int base;
    if (corner_cell(c, A, NN, S, base)) {
      const real_t ratio = cell[idx + base] / gn;
      acc += g_cell[idx + base] * ratio * ratio;
    }
  }
  g_stiff[idx] += share * acc;
}

} // extern "C"
