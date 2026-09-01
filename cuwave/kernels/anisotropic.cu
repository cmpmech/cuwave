// Compile-time configuration (set via -D flags from anisotropic.py):
//   USE_FLOAT
//   NDIM = 1 | 2 | 3
//   USE_DAMPING

#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

#ifndef RADIUS
#define RADIUS 1 // default order 2
#endif

// the node block a full-radius cell reaches, and the cells a node borders
#define BLK (2 * RADIUS)
#if NDIM == 1
#define CELLS BLK
#elif NDIM == 2
#define CELLS (BLK * BLK)
#else
#define CELLS (BLK * BLK * BLK)
#endif
#define NLOC (CELLS * NDIM)

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
#define AXIS_PARAMS const real_t f0, const int N0
#define INTERIOR_OR_RETURN                                                     \
  const int a0 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  if (!(a0 > 0 && a0 < N0 - 1))                                                \
    return;                                                                    \
  const int idx = a0
#define AXIS_GEOM const int A[1] = {a0}, S[1] = {1}, NN[1] = {N0}
#elif NDIM == 2
#define AXIS_PARAMS                                                            \
  const real_t f0, const int N0, const real_t f1, const int N1, const int s0
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
  const real_t f0, const int N0, const real_t f1, const int N1, const int s0,  \
      const real_t f2, const int N2, const int s1
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

// A cell sits inside the domain when both its own corners do. Its radius
// grades down towards a wall so the stencil never passes a ghost node, and
// the gathering node must fall inside that graded support to take from it.
__device__ __forceinline__ bool cell_inside(const int c, const int *A,
                                            const int *NN, const int *S,
                                            int &base, int &radius,
                                            int &self) {
  base = 0;
  self = 0;
  radius = RADIUS;
  bool inside = true;
  int power = 1;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int e = block_axis(c, d) - RADIUS; // low corner, from the node
    const int low = A[d] + e;
    inside &= (low >= 1 && low <= NN[d] - 3);
    radius = min(radius, min(RADIUS, min(low, NN[d] - 2 - low)));
    base += e * S[d];
    self += (RADIUS - 1 - e) * power;
    power *= BLK;
  }
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int e = block_axis(c, d) - RADIUS;
    inside &= (e >= -radius && e <= radius - 1);
  }
  return inside;
}

// offset of block entry l of cell c from the gathering node, and whether
// that entry falls inside the cell's graded support at all
__device__ __forceinline__ bool block_offset(const int c, const int l,
                                             const int radius, const int *S,
                                             int &off) {
  off = 0;
  bool carries = true;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int m = block_axis(l, d);
    carries &= (m >= RADIUS - radius && m <= RADIUS + radius - 1);
    off += (block_axis(c, d) - RADIUS + m - RADIUS + 1) * S[d];
  }
  return carries;
}

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ minv,
          const real_t *__restrict__ cell, const real_t *__restrict__ stencil,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
          const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;

  real_t force[NDIM];
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
    force[i] = (real_t)0;

#pragma unroll
  for (int c = 0; c < CELLS; ++c) {
    int base, radius, self;
    if (!cell_inside(c, A, NN, S, base, radius, self))
      continue;
    const real_t gc = cell[idx + base];
    const real_t *__restrict__ table = stencil + (radius - 1) * NLOC * NLOC;
#pragma unroll
    for (int l = 0; l < CELLS; ++l) {
      int off;
      if (!block_offset(c, l, radius, S, off))
        continue;
#pragma unroll
      for (int j = 0; j < NDIM; ++j) {
        const real_t uj = u1[j * cs + idx + off];
#pragma unroll
        for (int i = 0; i < NDIM; ++i)
          force[i] -=
              gc * table[(self * NDIM + i) * NLOC + l * NDIM + j] * uj;
      }
    }
  }

  const real_t mi = minv[idx];
#ifdef USE_DAMPING
  const real_t beta = 0.5f * mi * damping[idx] * dt;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    const int n = i * cs + idx;
    u2[n] = (2.f * u1[n] - u0[n] * (1.f - beta) + f0 * mi * force[i]) /
            (1.f + beta);
  }
#else
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    const int n = i * cs + idx;
    u2[n] = -u0[n] + 2.f * u1[n] + f0 * mi * force[i];
  }
#endif
}

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

} // extern "C"
