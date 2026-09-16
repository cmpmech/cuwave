// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 2 | 3
//   USE_DAMPING
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

// ------------------------------------ curl helpers
// the (k, l) curl component at its own staggered point, the elastic shear
// strain with its first term negated: both taps are half-point differences
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

// cell weight of a pair point, halved on the walls of the axes it does not span
__device__ __forceinline__ real_t pair_weight(const int k, const int l,
                                              const int *A, const int *NN) {
  real_t w = (real_t)1;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    if (d != k && d != l && (A[d] == 1 || A[d] == NN[d] - 2))
      w *= (real_t)0.5;
  return w;
}

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void curl_kernel(const real_t *__restrict__ u1,
                            real_t *__restrict__ h,
#ifdef USE_MAGNETIC
                            const real_t *__restrict__ nu_pair,
#endif
                            const real_t nu, const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      if (A[k] > NN[k] - 3 || A[l] > NN[l] - 3)
        continue; // no pair point on the staggered ghost of either axis
      const int p = PAIR_ROW(k, l) - NDIM;
      const real_t b = curl_component(u1, idx, cs, k, l, A, S, NN, F);
#ifdef USE_MAGNETIC
      h[p * cs + idx] = nu_pair[p * cs + idx] * b; // the weight is folded in
#else
      h[p * cs + idx] = nu * pair_weight(k, l, A, NN) * b;
#endif
    }
}

// ------------------------------------------------------------------------------------
__global__ void
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ h,
          const real_t *__restrict__ minv,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
          const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    real_t force = (real_t)0;
#pragma unroll
    for (int m = 0; m < NDIM; ++m) { // the pairs this component belongs to
      if (m == c)
        continue;
      const int p = PAIR_ROW(min(c, m), max(c, m)) - NDIM;
      const real_t *__restrict__ hp = h + p * cs;
      // E_c enters its pair as +d_m E_c for m < c and as -d_m E_c for m > c
      const real_t sgn = (m < c) ? (real_t)1 : (real_t)-1;
      real_t acc = (real_t)0;
#pragma unroll
      for (int j = 1; j <= STENCIL_RADIUS; ++j) {
        const int ap = A[m] + j - 1;
        const int rp = (ap <= NN[m] - 3) ? rad_half(ap, NN[m]) : 0;
        if (j <= rp)
          acc += SG_W(rp, j) * hp[idx + (j - 1) * S[m]];
        const int am = A[m] - j;
        const int rm = (am >= 1) ? rad_half(am, NN[m]) : 0;
        if (j <= rm)
          acc -= SG_W(rm, j) * hp[idx - j * S[m]];
      }
      force += sgn * F[m] * acc;
    }
    const int n = c * cs + idx;
    const real_t mi = minv[n];
#ifdef USE_DAMPING
    const real_t beta = 0.5f * mi * damping[idx] * dt;
    u2[n] = (2.f * u1[n] - u0[n] * (1.f - beta) + mi * force) / (1.f + beta);
#else
    u2[n] = -u0[n] + 2.f * u1[n] + mi * force;
#endif
  }
}

} // extern "C"
