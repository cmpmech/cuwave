// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 2 | 3
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
__device__ __forceinline__ real_t
curl_component(const real_t *__restrict__ u, const int idx, const int cs,
               const int k, const int l, const int *A, const int *S,
               const int *NN, const real_t *F) {
  const int r1 = rad_half(A[l], NN[l]); // d u_k / d x_l
  const int r2 = rad_half(A[k], NN[k]); // d u_l / d x_k
  real_t b = (real_t)0;
#pragma unroll
  for (int j = 1; j <= STENCIL_RADIUS; ++j) {
    if (j <= r1)
      b -= F[l] * SG_W(r1, j) *
           (u[k * cs + idx + j * S[l]] - u[k * cs + idx - (j - 1) * S[l]]);
    if (j <= r2)
      b += F[k] * SG_W(r2, j) *
           (u[l * cs + idx + j * S[k]] - u[l * cs + idx - (j - 1) * S[k]]);
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
