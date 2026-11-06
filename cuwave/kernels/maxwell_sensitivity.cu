// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 2 | 3
//   CONDUCTOR_FACES
//   USE_MAGNETIC

#if NDIM == 1
#error "a single in-plane component has no curl; use maxwell.ElectricWave"
#endif

#if NDIM == 2
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
#define AXIS_FACTORS const real_t F[2] = {f0, f1}
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
#define AXIS_FACTORS const real_t F[3] = {f0, f1, f2}
#endif

#ifdef USE_MAGNETIC
// ------------------------------------ curl helpers
// byte-identical to maxwell.cu: each file is its own compilation unit
#ifndef CONDUCTOR_FACES
#define CONDUCTOR_FACES 0
#endif

// the (k, l) curl component at its own staggered point, the elastic shear
// strain with its first term negated: both taps are half-point differences of
// a tangential component, read past a wall off its image, odd at a conductor
__device__ __forceinline__ real_t
curl_component(const real_t *__restrict__ u, const int idx, const int cs,
               const int k, const int l, const int *A, const int *S,
               const int *NN, const real_t *F) {
  // u_k along l and u_l along k, nodes 1 - R .. R away
  real_t tk[2 * STENCIL_RADIUS], tl[2 * STENCIL_RADIUS];
  staged_taps<false>(tk, u + k * cs, idx, S[l], A[l], NN[l],
                     CONDUCTOR_FACES >> 2 * l, 1 - STENCIL_RADIUS, 1,
                     NN[l] - 2);
  staged_taps<false>(tl, u + l * cs, idx, S[k], A[k], NN[k],
                     CONDUCTOR_FACES >> 2 * k, 1 - STENCIL_RADIUS, 1,
                     NN[k] - 2);
  real_t b = (real_t)0;
#pragma unroll
  for (int j = 1; j <= STENCIL_RADIUS; ++j) {
    b -= F[l] * SG_W(STENCIL_RADIUS, j) *
         (tk[STENCIL_RADIUS - 1 + j] - tk[STENCIL_RADIUS - j]);
    b += F[k] * SG_W(STENCIL_RADIUS, j) *
         (tl[STENCIL_RADIUS - 1 + j] - tl[STENCIL_RADIUS - j]);
  }
  return b;
}
#endif

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void gradient_kernel(real_t *__restrict__ g_mass,
#ifdef USE_MAGNETIC
                                real_t *__restrict__ g_nu,
#endif
                                const real_t *__restrict__ u0,
                                const real_t *__restrict__ u1,
                                const real_t *__restrict__ u2,
                                const real_t *__restrict__ l1, const real_t mf,
                                const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;

  // dJ/dpermittivity on the component points: no neighbour and no material load
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue;
    const int n = c * cs + idx;
    g_mass[n] -= mf * l1[n] * (u2[n] - 2.f * u1[n] + u0[n]);
  }

#ifdef USE_MAGNETIC
  AXIS_FACTORS;
  // dJ/d(1/mu) on the pair points: forward curl against adjoint curl
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      if (A[k] > NN[k] - 3 || A[l] > NN[l] - 3)
        continue;
      const int p = PAIR_ROW(k, l) - NDIM;
      g_nu[p * cs + idx] -= curl_component(l1, idx, cs, k, l, A, S, NN, F) *
                            curl_component(u1, idx, cs, k, l, A, S, NN, F);
    }
#endif
}

// ------------------------------------------------------------------------------------
__global__ void frechet_kernel(real_t *__restrict__ acc_mass,
#ifdef USE_MAGNETIC
                               real_t *__restrict__ acc_nu,
#endif
                               const real_t *__restrict__ u0,
                               const real_t *__restrict__ u1,
                               const real_t *__restrict__ u2, const real_t ft,
                               const real_t fs, const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;

#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue;
    const int n = c * cs + idx;
    const real_t dudt = u2[n] - u0[n];
    acc_mass[n] += ft * dudt * dudt;
  }

#ifdef USE_MAGNETIC
  AXIS_FACTORS;
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      if (A[k] > NN[k] - 3 || A[l] > NN[l] - 3)
        continue;
      const int p = PAIR_ROW(k, l) - NDIM;
      const real_t b = curl_component(u1, idx, cs, k, l, A, S, NN, F);
      acc_nu[p * cs + idx] += fs * b * b;
    }
#endif
}

} // extern "C"
