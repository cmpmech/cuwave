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
