// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3

#if NDIM == 1
#define AXIS_PARAMS const real_t f0, const int N0
#define INTERIOR_OR_RETURN                                                     \
  const int a0 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  if (!(a0 > 0 && a0 < N0 - 1))                                                \
    return;                                                                    \
  const int idx = a0
#define AXIS_GEOM const int A[1] = {a0}, S[1] = {1}, NN[1] = {N0}
#define AXIS_FACTORS const real_t F[1] = {f0}
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

// the node's normal strains, one per axis: a traction wall's is a zero row
// (returned as a set bit), a clamped one's reads its held u = 0 as nothing
template <int DEEP = 0>
__device__ __forceinline__ int
normal_strains(const real_t *__restrict__ u, const int idx, const int cs,
               const int *A, const int *S, const int *NN, const real_t *F,
               const int clamped, real_t *eps) {
  int zeroed = 0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const bool deep = (DEEP >> d) & 1;
    if (!deep && ((A[d] == 1 && !clamped_face(clamped, d, 0)) ||
                  (A[d] == NN[d] - 2 && !clamped_face(clamped, d, 1)))) {
      eps[d] = (real_t)0;
      zeroed |= 1 << d;
    } else {
      eps[d] = F[d] * node_derivative(Line{u + d * cs, idx, S[d]}, A[d], NN[d],
                                      deep);
    }
  }
  return zeroed;
}

// a traction wall condenses its axis out of the coupling: the plane stress
// reduction of lam, applied once per zeroed axis
__device__ __forceinline__ real_t condensed_lame(const real_t lam,
                                                 const real_t mu,
                                                 const int zeroed) {
  real_t lam_eff = lam;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    if ((zeroed >> d) & 1)
      lam_eff = 2.f * lam_eff * mu / (lam_eff + 2.f * mu);
  return lam_eff;
}

#if NDIM >= 2
// the engineering shear strain of the (k, l) pair at its own staggered point,
// rounded as the step's
template <int DEEP = 0>
__device__ __forceinline__ real_t
shear_strain(const real_t *__restrict__ u, const int idx, const int cs,
             const int k, const int l, const int *A, const int *S,
             const int *NN, const real_t *F) {
  const real_t dk = half_derivative(Line{u + l * cs, idx, S[k]}, A[k], NN[k],
                                    (DEEP >> k) & 1);
  const real_t dl = half_derivative(Line{u + k * cs, idx, S[l]}, A[l], NN[l],
                                    (DEEP >> l) & 1);
  return FMA_RN(F[k], dk, MUL_RN(F[l], dl));
}
#endif

// the normal bilinear form of two strain sets under the condensed coupling
__device__ __forceinline__ real_t normal_form(const real_t *ea,
                                              const real_t *eb,
                                              const real_t lam,
                                              const real_t mu,
                                              const int zeroed) {
  const real_t lam_eff = condensed_lame(lam, mu, zeroed);
  real_t tra = (real_t)0, trb = (real_t)0, dot = (real_t)0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    tra += ea[d];
    trb += eb[d];
    dot += ea[d] * eb[d];
  }
  return 2.f * mu * dot + lam_eff * tra * trb;
}

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void gradient_kernel(real_t *__restrict__ g_mass,
                                real_t *__restrict__ g_normal,
#if NDIM >= 2
                                real_t *__restrict__ g_shear,
#endif
                                const real_t *__restrict__ u0,
                                const real_t *__restrict__ u1,
                                const real_t *__restrict__ u2,
                                const real_t *__restrict__ l1, const real_t lam,
                                const real_t mu, const real_t mf,
                                const int clamped, const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

  // dJ/dmass on the component points: no neighbour and no material load
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue;
    const int n = c * cs + idx;
    g_mass[n] -= mf * l1[n] * (u2[n] - 2.f * u1[n] + u0[n]);
  }

  // dJ/dgamma on the node: forward against adjoint under the normal coupling,
  // off the walls on the deep path, which rounds as the general one there
  const int deep = warp_deep(A, NN);
  real_t eu[NDIM], el[NDIM];
  const int zeroed =
      BY_DEEP(deep, normal_strains, u1, idx, cs, A, S, NN, F, clamped, eu);
  BY_DEEP(deep, normal_strains, l1, idx, cs, A, S, NN, F, clamped, el);
  g_normal[idx] -= normal_form(el, eu, lam, mu, zeroed);

#if NDIM >= 2
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k) // and on each of its shear points
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      if (A[k] > NN[k] - 3 || A[l] > NN[l] - 3)
        continue;
      const int v = PAIR_ROW(k, l);
      g_shear[(v - NDIM) * cs + idx] -=
          mu * BY_DEEP(deep, shear_strain, l1, idx, cs, k, l, A, S, NN, F) *
          BY_DEEP(deep, shear_strain, u1, idx, cs, k, l, A, S, NN, F);
    }
#endif
}

// ------------------------------------------------------------------------------------
__global__ void frechet_kernel(real_t *__restrict__ acc_mass,
                               real_t *__restrict__ acc_normal,
#if NDIM >= 2
                               real_t *__restrict__ acc_shear,
#endif
                               const real_t *__restrict__ u0,
                               const real_t *__restrict__ u1,
                               const real_t *__restrict__ u2, const real_t lam,
                               const real_t mu, const real_t ft,
                               const real_t fs, const int clamped,
                               const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue;
    const int n = c * cs + idx;
    const real_t dudt = u2[n] - u0[n];
    acc_mass[n] += ft * dudt * dudt;
  }

  const int deep = warp_deep(A, NN);
  real_t eu[NDIM];
  const int zeroed =
      BY_DEEP(deep, normal_strains, u1, idx, cs, A, S, NN, F, clamped, eu);
  acc_normal[idx] += fs * normal_form(eu, eu, lam, mu, zeroed);

#if NDIM >= 2
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      if (A[k] > NN[k] - 3 || A[l] > NN[l] - 3)
        continue;
      const int v = PAIR_ROW(k, l);
      const real_t e =
          BY_DEEP(deep, shear_strain, u1, idx, cs, k, l, A, S, NN, F);
      acc_shear[(v - NDIM) * cs + idx] += fs * mu * e * e;
    }
#endif
}

} // extern "C"
