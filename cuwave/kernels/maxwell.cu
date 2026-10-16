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

// ------------------------------------- pml helpers
// the curl row of the pair (c, e), counted from the first pair
__device__ __forceinline__ int curl_row(const int c, const int e) {
  return PAIR_ROW(min(c, e), max(c, e)) - NDIM;
}

// curl (c, e) j nodes along axis e from a node, what component c diverges
struct Curl {
  const real_t *__restrict__ p;
  int idx, cs, step, e;
  __device__ real_t operator()(const int c, const int j) const {
    return p[curl_row(c, e) * cs + idx + j * step];
  }
};

// the signed d E_k / d x_e at the pair point (k, e), as curl_component takes it
template <class F>
__device__ __forceinline__ real_t curl_term(const F &u, const int k,
                                            const int e, const int *A,
                                            const int *NN, const real_t fe) {
  if (k == e || A[k] > NN[k] - 3 || A[e] > NN[e] - 3)
    return (real_t)0; // no pair point there
  const int r = rad_half(A[e], NN[e]);
  real_t acc = (real_t)0;
#pragma unroll
  for (int j = 1; j <= STENCIL_RADIUS; ++j)
    if (j <= r)
      acc += SG_W(r, j) * (u(k, j) - u(k, -(j - 1)));
  return (k < e ? -fe : fe) * acc; // the first term of the pair is negated
}

// the axis-e part of the force on component c, as fd_kernel takes it
template <class F>
__device__ __forceinline__ real_t divergence_term(const F &h, const int c,
                                                  const int e, const int *A,
                                                  const int *NN,
                                                  const real_t fe) {
  real_t acc = (real_t)0;
#pragma unroll
  for (int j = 1; j <= STENCIL_RADIUS; ++j) {
    const int ap = A[e] + j - 1;
    const int rp = (ap <= NN[e] - 3) ? rad_half(ap, NN[e]) : 0;
    if (j <= rp)
      acc += SG_W(rp, j) * h(c, j - 1);
    const int am = A[e] - j, rm = (am >= 1) ? rad_half(am, NN[e]) : 0;
    if (j <= rm)
      acc -= SG_W(rm, j) * h(c, -j);
  }
  return (e < c ? fe : -fe) * acc;
}

// nu times the curl terms t of axis e, added onto the pair points
__device__ __forceinline__ void add_curl(real_t *__restrict__ h,
#ifdef USE_MAGNETIC
                                         const real_t *__restrict__ nu_pair,
#endif
                                         const real_t nu, const int idx,
                                         const int cs, const int e,
                                         const real_t *t, const int *A,
                                         const int *NN) {
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
    if (k != e && A[k] <= NN[k] - 3 && A[e] <= NN[e] - 3) {
      const int p = curl_row(k, e);
#ifdef USE_MAGNETIC
      h[p * cs + idx] += nu_pair[p * cs + idx] * t[k];
#else
      h[p * cs + idx] += nu * pair_weight(min(k, e), max(k, e), A, NN) * t[k];
#endif
    }
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

// ------------------------------------------------------------------------------------
__global__ void pml_curl_kernel(const real_t *__restrict__ u1,
                                real_t *__restrict__ h,
#ifdef USE_MAGNETIC
                                const real_t *__restrict__ nu_pair,
#endif
                                const real_t nu, const int cs,
                                real_t *__restrict__ aux,
                                const real_t *__restrict__ profile, const int e,
                                const int lo, const int hi, AXIS_PARAMS) {
  PML_AXES;
  SLAB_OR_RETURN(0);
  const Field u = {u1, idx, cs, S[e]};
  real_t psi[NDIM];
#pragma unroll
  for (int k = 0; k < NDIM; ++k) {
    const int at = 2 * NN[e] + a; // every pair point sits half a node up e
    psi[k] = profile[NN[e] + at] * aux[k * size + t] +
             profile[at] * curl_term(u, k, e, A, NN, F[e]);
    aux[k * size + t] = psi[k];
  }
  add_curl(h,
#ifdef USE_MAGNETIC
           nu_pair,
#endif
           nu, idx, cs, e, psi, A, NN);
}

// ------------------------------------------------------------------------------------
__global__ void
pml_force_kernel(const real_t *__restrict__ u1, real_t *__restrict__ u2,
                 const real_t *__restrict__ h, const real_t *__restrict__ minv,
                 const int cs, real_t *__restrict__ aux,
                 const real_t *__restrict__ profile, const int e, const int lo,
                 const int hi, AXIS_PARAMS) {
  // u1 unread: the adjoint's signature, so one launch list serves both
  PML_AXES;
  SLAB_OR_RETURN(0);
  const Curl curl = {h, idx, cs, S[e], e};
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (c == e || A[c] > NN[c] - 3)
      continue; // no axis-e curl on its own component, no unknown on a ghost
    real_t *z = aux + (NDIM + c) * size + t; // E_c sits on the node along e
    *z = profile[NN[e] + a] * *z +
         profile[a] * divergence_term(curl, c, e, A, NN, F[e]);
    u2[c * cs + idx] += minv[c * cs + idx] * *z;
  }
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_curl_kernel(const real_t *__restrict__ l1,
                                        real_t *__restrict__ h,
#ifdef USE_MAGNETIC
                                        const real_t *__restrict__ nu_pair,
#endif
                                        const real_t nu, const int cs,
                                        const real_t *__restrict__ aux,
                                        const real_t *__restrict__ profile,
                                        const int e, const int lo, const int hi,
                                        AXIS_PARAMS) {
  PML_AXES;
  SLAB_OR_RETURN(STENCIL_RADIUS);
  // the force memory pushed back through the curl: the force's transpose
  const Memory<Field> m = {{l1, idx, cs, S[e]},
                           aux,
                           profile,
                           NN,
                           size,
                           NDIM,
                           a,
                           off,
                           e,
                           lo,
                           hi,
                           plane,
                           true};
  real_t q[NDIM];
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
    q[k] = curl_term(m, k, e, A, NN, F[e]);
  add_curl(h,
#ifdef USE_MAGNETIC
           nu_pair,
#endif
           nu, idx, cs, e, q, A, NN);
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_force_kernel(
    const real_t *__restrict__ l1, real_t *__restrict__ l2,
    const real_t *__restrict__ h, const real_t *__restrict__ minv, const int cs,
    real_t *__restrict__ aux, const real_t *__restrict__ profile, const int e,
    const int lo, const int hi, AXIS_PARAMS) {
  PML_AXES;
  SLAB_OR_RETURN(STENCIL_RADIUS);
  // the curl memory pushed back through the force: the curl's transpose
  const Memory<Curl> m = {{h, idx, cs, S[e], e},
                          aux,
                          profile,
                          NN,
                          size,
                          0,
                          a,
                          off,
                          e,
                          lo,
                          hi,
                          plane,
                          false};
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (c == e || A[c] > NN[c] - 3)
      continue;
    const int n = c * cs + idx;
    l2[n] += minv[n] * divergence_term(m, c, e, A, NN, F[e]);
    if (t >= 0) { // the force memory, read by the curl pass before
      real_t *z = aux + (NDIM + c) * size + t;
      *z = profile[NN[e] + a] * (*z + l1[n]);
    }
  }
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_memory_kernel(const real_t *__restrict__ h,
                                          const int cs,
                                          real_t *__restrict__ aux,
                                          const real_t *__restrict__ profile,
                                          const int e, const int lo,
                                          const int hi, AXIS_PARAMS) {
  PML_AXES;
  SLAB_OR_RETURN(0);
#pragma unroll
  for (int k = 0; k < NDIM; ++k) {
    if (k == e)
      continue;
    real_t *m = aux + k * size + t;
    *m = profile[3 * NN[e] + a] * (*m + h[curl_row(k, e) * cs + idx]);
  }
}

} // extern "C"
