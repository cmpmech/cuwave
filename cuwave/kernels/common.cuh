// Prelude prepended to every .cu by compile_kernels, after the stencil table.
// Compile-time configuration (set via -D flags and stencils.preamble):
//   USE_FLOAT
//   NDIM = 1 | 2 | 3
//   STENCIL_RADIUS
//   OP_COEFFS
//   STAG_COEFFS

#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

// ----------------------------- finite difference helpers
#ifndef STENCIL_RADIUS
#define STENCIL_RADIUS 1 // default order 2
#endif

#if STENCIL_RADIUS == 1
#define OP_W(r, k) ((real_t)1)
#define CLOSURE(a, N) 1
#define SG_W(r, k) ((real_t)1)
#else
// cache
__constant__ real_t OP_C[STENCIL_RADIUS][STENCIL_RADIUS] = OP_COEFFS;
#define OP_W(r, k) OP_C[(r) - 1][(k) - 1] // 1-indexed adjustment
// grade radius down towards wall so the stencil never reaches past ghost nodes
#define CLOSURE(a, N) min(STENCIL_RADIUS, min(a, (N) - 1 - (a)))
__constant__ real_t SG_C[STENCIL_RADIUS][STENCIL_RADIUS] = STAG_COEFFS;
#define SG_W(r, k) SG_C[(r) - 1][(k) - 1] // 1-indexed adjustment
#endif

// --------------------------------- staggered lattice
#define NPAIRS (NDIM * (NDIM - 1) / 2)
#define NVOIGT (NDIM + NPAIRS)
#if NDIM == 3
#define PAIR_ROW(k, l) (NDIM + 3 - (k) - (l))
#else
#define PAIR_ROW(k, l) 2
#endif

// radius of the node-centred derivative of a staggered field, zero on the wall
__device__ __forceinline__ int rad_node(const int a, const int N) {
  return min(STENCIL_RADIUS, min(a - 1, N - 2 - a));
}

// radius of a half-point derivative, whose taps may sit on the wall
__device__ __forceinline__ int rad_half(const int a, const int N) {
  return min(STENCIL_RADIUS, min(a, N - 2 - a));
}

__device__ __forceinline__ bool clamped_face(const int clamped, const int d,
                                             const int side) {
  return (clamped >> (2 * d + side)) & 1;
}

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
