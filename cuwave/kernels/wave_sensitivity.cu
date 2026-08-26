// Compile-time configuration (set via -D flags from sensitivity.py):
//   USE_FLOAT
//   NDIM = 1 | 2 | 3
//   STENCIL_RADIUS
//   OP_COEFFS

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
#else
// cache
__constant__ real_t OP_C[STENCIL_RADIUS][STENCIL_RADIUS] = OP_COEFFS;
#define OP_W(r, k) OP_C[(r) - 1][(k) - 1] // 1-indexed adjustment
// grade radius down towards wall so the stencil never reaches past ghost nodes
#define CLOSURE(a, N) min(STENCIL_RADIUS, min(a, (N) - 1 - (a)))
#endif

// ------------------------------- interior guard macros
#if NDIM == 1
#define INTERIOR_OR_RETURN                                                     \
  const int a0 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  if (!(a0 > 0 && a0 < N0 - 1))                                                \
    return;                                                                    \
  const int idx = a0
#define AXIS_RADII const int r0 = CLOSURE(a0, N0)
#define AXIS_OFFSETS const int o0 = 1
#elif NDIM == 2
#define INTERIOR_OR_RETURN                                                     \
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;                        \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1))                       \
    return;                                                                    \
  const int idx = a0 * s0 + a1
#define AXIS_RADII const int r0 = CLOSURE(a0, N0), r1 = CLOSURE(a1, N1)
#define AXIS_OFFSETS const int o0 = s0, o1 = 1
#elif NDIM == 3
#define INTERIOR_OR_RETURN                                                     \
  const int a2 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  const int a1 = blockIdx.y * blockDim.y + threadIdx.y;                        \
  const int a0 = blockIdx.z * blockDim.z + threadIdx.z;                        \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&            \
        a2 < N2 - 1))                                                          \
    return;                                                                    \
  const int idx = a0 * s0 + a1 * s1 + a2
#define AXIS_RADII                                                             \
  const int r0 = CLOSURE(a0, N0), r1 = CLOSURE(a1, N1), r2 = CLOSURE(a2, N2)
#define AXIS_OFFSETS const int o0 = s0, o1 = s1, o2 = 1
#endif

// ---------------------------------- gradient helpers
__device__ __forceinline__ real_t stiffness_gradient_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ l1,
    const real_t *__restrict__ stiff, const int idx, const int s,
    const real_t uc, const real_t lc, const real_t sc, const real_t factor,
    const int r) {
  const real_t sp = stiff[idx + s];                     // plus of sc
  const real_t sm = stiff[idx - s];                     // minus of sc
  const real_t dgp = sp * sp / ((sc + sp) * (sc + sp)); // d(harmonic mean)/dsc
  const real_t dgm = sm * sm / ((sc + sm) * (sc + sm)); // d(harmonic mean)/dsc
  real_t Dp = OP_W(r, 1) * (u1[idx + s] - uc);          // inner grad
  real_t Dm = OP_W(r, 1) * (uc - u1[idx - s]);          // inner grad
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Dp += OP_W(r, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]);
      Dm += OP_W(r, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]);
    }
  return factor * (dgp * Dp * (l1[idx + s] - lc) +
                   dgm * Dm * (lc - l1[idx - s])); // both faces of the node
}

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
gradient_kernel(real_t *__restrict__ g_mass, real_t *__restrict__ g_stiff,
                const real_t *__restrict__ u0, const real_t *__restrict__ u1,
                const real_t *__restrict__ u2, const real_t *__restrict__ l1,
                const real_t *__restrict__ stiff, const real_t inv_dt2,
                const real_t F0, const int N0
#if NDIM >= 2
                ,
                const real_t F1, const int N1, const int s0
#endif
#if NDIM >= 3
                ,
                const real_t F2, const int N2, const int s1
#endif
) {
  INTERIOR_OR_RETURN;
  AXIS_RADII;

  const real_t uc = u1[idx];
  const real_t lc = l1[idx];

  // dJ/dmass: no neighbour and no material load
  g_mass[idx] -= inv_dt2 * lc * (u2[idx] - 2.f * uc + u0[idx]);

  // dJ/dstiff: wave.cu's harmonic face mean differentiated in place
  const real_t sc = stiff[idx];
#if NDIM == 1
  const real_t g =
      stiffness_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, F0, r0);
#elif NDIM == 2
  const real_t g =
      stiffness_gradient_axis(u1, l1, stiff, idx, s0, uc, lc, sc, F0, r0) +
      stiffness_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, F1, r1);
#elif NDIM == 3
  const real_t g =
      stiffness_gradient_axis(u1, l1, stiff, idx, s0, uc, lc, sc, F0, r0) +
      stiffness_gradient_axis(u1, l1, stiff, idx, s1, uc, lc, sc, F1, r1) +
      stiffness_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, F2, r2);
#endif
  g_stiff[idx] -= g;
}

// ------------------------------------------------------------------------------------
__global__ void frechet_kernel(real_t *__restrict__ acc_mass,
                               real_t *__restrict__ acc_stiff,
                               const real_t *__restrict__ u0,
                               const real_t *__restrict__ u1,
                               const real_t *__restrict__ u2, const real_t ft,
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
  INTERIOR_OR_RETURN;
  AXIS_OFFSETS;

  const real_t dudt = u2[idx] - u0[idx];
  acc_mass[idx] += ft * dudt * dudt;

  const real_t g0 = u1[idx + o0] - u1[idx - o0];
  real_t sum = f0 * g0 * g0;
#if NDIM >= 2
  const real_t g1 = u1[idx + o1] - u1[idx - o1];
  sum += f1 * g1 * g1;
#endif
#if NDIM >= 3
  const real_t g2 = u1[idx + o2] - u1[idx - o2];
  sum += f2 * g2 * g2;
#endif
  acc_stiff[idx] += sum;
}

} // extern "C"
