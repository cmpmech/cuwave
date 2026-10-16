// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3
//   USE_DAMPING

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

// a node 2R inside every wall: full radius on every tap, no wall terms
__device__ __forceinline__ bool deep_node(const int *A, const int *NN) {
  bool deep = true;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    deep &= A[d] > 2 * STENCIL_RADIUS && A[d] < NN[d] - 2 - 2 * STENCIL_RADIUS;
  return deep;
}

template <bool DEEP>
__device__ __forceinline__ int node_radius(const int a, const int N) {
  return DEEP ? STENCIL_RADIUS : rad_node(a, N);
}

template <bool DEEP>
__device__ __forceinline__ int half_radius(const int a, const int N) {
  return DEEP ? STENCIL_RADIUS : rad_half(a, N);
}

template <bool DEEP>
__device__ __forceinline__ bool clamped_at(const int clamped, const int d,
                                           const int side) {
  return !DEEP && clamped_face(clamped, d, side);
}

// the node's normal strains, one per axis: a wall grades to a zero row under
// traction (returned as a set bit) and to the antisymmetric fold when clamped
template <bool DEEP = false>
__device__ __forceinline__ int
normal_strains(const real_t *__restrict__ u, const int idx, const int cs,
               const int *A, const int *S, const int *NN, const real_t *F,
               const int clamped, real_t *eps) {
  int zeroed = 0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int r = node_radius<DEEP>(A[d], NN[d]);
    const real_t *__restrict__ ud = u + d * cs;
    if (r > 0) {
      real_t acc = (real_t)0;
#pragma unroll
      for (int k = 1; k <= STENCIL_RADIUS; ++k)
        if (k <= r)
          acc += SG_W(r, k) * (ud[idx + (k - 1) * S[d]] - ud[idx - k * S[d]]);
      eps[d] = F[d] * acc;
    } else if (A[d] == 1 && clamped_at<DEEP>(clamped, d, 0)) {
      eps[d] = F[d] * 2.f * ud[idx]; // the wall holds u = 0 half a node down
    } else if (A[d] == NN[d] - 2 && clamped_at<DEEP>(clamped, d, 1)) {
      eps[d] = -F[d] * 2.f * ud[idx - S[d]];
    } else {
      eps[d] = (real_t)0;
      zeroed |= 1 << d;
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
// the engineering shear strain of the (k, l) pair at its own staggered point
template <bool DEEP = false>
__device__ __forceinline__ real_t
shear_strain(const real_t *__restrict__ u, const int idx, const int cs,
             const int k, const int l, const int *A, const int *S,
             const int *NN, const real_t *F) {
  const int r1 = half_radius<DEEP>(A[l], NN[l]); // d u_k / d x_l
  const int r2 = half_radius<DEEP>(A[k], NN[k]); // d u_l / d x_k
  real_t e = (real_t)0;
#pragma unroll
  for (int j = 1; j <= STENCIL_RADIUS; ++j) {
    if (j <= r1)
      e += F[l] * SG_W(r1, j) *
           (u[k * cs + idx + j * S[l]] - u[k * cs + idx - (j - 1) * S[l]]);
    if (j <= r2)
      e += F[k] * SG_W(r2, j) *
           (u[l * cs + idx + j * S[k]] - u[l * cs + idx - (j - 1) * S[k]]);
  }
  return e;
}
#endif

// ---------------------------------- material helpers
#ifdef USE_FLOAT
#define TINY 1.17549435e-38f // finfo.tiny, the guard the python averages take
#else
#define TINY 2.2250738585072014e-308
#endif

// the cell weight of a node, halved on the walls of every axis outside `skip`
template <bool DEEP = false>
__device__ __forceinline__ real_t wall_weight(const int *A, const int *NN,
                                              const int skip) {
  real_t w = (real_t)1;
  if (DEEP)
    return w;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    if (!((skip >> d) & 1)) {
      if (A[d] == 1)
        w *= (real_t)0.5;
      if (A[d] == NN[d] - 2)
        w *= (real_t)0.5;
    }
  return w;
}

// the design on the normal stress point, the node itself
template <bool DEEP = false>
__device__ __forceinline__ real_t node_gamma(const real_t *__restrict__ gamma,
                                             const int idx, const int *A,
                                             const int *NN) {
  return wall_weight<DEEP>(A, NN, 0) * gamma[idx];
}

#if NDIM >= 2
// the design on the (k, l) shear point: harmonic over the four nodes around it
template <bool DEEP = false>
__device__ __forceinline__ real_t
shear_gamma(const real_t *__restrict__ gamma, const int idx, const int k,
            const int l, const int *A, const int *S, const int *NN) {
  real_t sum = (real_t)1 / fmax(gamma[idx], TINY);
  sum += (real_t)1 / fmax(gamma[idx + S[l]], TINY);
  sum += (real_t)1 / fmax(gamma[idx + S[k]], TINY);
  sum += (real_t)1 / fmax(gamma[idx + S[k] + S[l]], TINY);
  return wall_weight<DEEP>(A, NN, (1 << k) | (1 << l)) * ((real_t)4 / sum);
}
#endif

// inverse inertia of component c's point, zero where a clamped wall holds it
template <bool DEEP = false>
__device__ __forceinline__ real_t
point_minv(const real_t *__restrict__ gamma, const int idx, const int c,
           const int *A, const int *S, const int *NN, const real_t rho,
           const int clamped) {
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    if (d != c && ((A[d] == 1 && clamped_at<DEEP>(clamped, d, 0)) ||
                   (A[d] == NN[d] - 2 && clamped_at<DEEP>(clamped, d, 1))))
      return (real_t)0;
  const real_t mean = (real_t)0.5 * (gamma[idx] + gamma[idx + S[c]]);
  return (real_t)1 / fmax(rho * wall_weight<DEEP>(A, NN, 1 << c) * mean, TINY);
}

// ----------------------------------- step helpers
// every stress of the node at idx, zero off its point
template <bool DEEP = false>
__device__ __forceinline__ void
stress_point(const real_t *__restrict__ u, const real_t *__restrict__ gamma,
             const int idx, const int cs, const int *A, const int *S,
             const int *NN, const real_t *F, const real_t lam, const real_t mu,
             const int clamped, real_t *sig) {
  real_t eps[NDIM];
  const int zeroed =
      normal_strains<DEEP>(u, idx, cs, A, S, NN, F, clamped, eps);
  const real_t lam_eff = condensed_lame(lam, mu, zeroed);
  real_t tr = (real_t)0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    tr += eps[d];
  const real_t gn = node_gamma<DEEP>(gamma, idx, A, NN);
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    sig[d] = ((zeroed >> d) & 1) ? (real_t)0
                                 : gn * (2.f * mu * eps[d] + lam_eff * tr);

#if NDIM >= 2
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      const int v = PAIR_ROW(k, l);
      sig[v] = (!DEEP && (A[k] > NN[k] - 3 || A[l] > NN[l] - 3))
                   ? (real_t)0
                   : shear_gamma<DEEP>(gamma, idx, k, l, A, S, NN) * mu *
                         shear_strain<DEEP>(u, idx, cs, k, l, A, S, NN, F);
    }
#endif
}

// the stress divergence on component c's point, off the accessor (sigma, cs, S)
template <bool DEEP = false>
__device__ __forceinline__ real_t
nodal_force(const real_t *__restrict__ sigma, const int idx, const int cs,
            const int *S, const int c, const int *A, const int *NN,
            const real_t *F, const int clamped) {
  real_t force = (real_t)0;
  { // own normal stress along axis c, every tap keyed off the stress node
    const real_t *__restrict__ sc = sigma + c * cs;
    real_t acc = (real_t)0;
#pragma unroll
    for (int k = 1; k <= STENCIL_RADIUS; ++k) {
      const int ap = A[c] + k;
      const int rp = node_radius<DEEP>(ap, NN[c]);
      if (k <= rp)
        acc += SG_W(rp, k) * sc[idx + k * S[c]];
      else if (k == 1 && ap == NN[c] - 2 && clamped_at<DEEP>(clamped, c, 1))
        acc += 2.f * sc[idx + S[c]]; // the fold of the clamped wall strain
      const int am = A[c] - k + 1;
      const int rm = (DEEP || am >= 1) ? node_radius<DEEP>(am, NN[c]) : 0;
      if (k <= rm)
        acc -= SG_W(rm, k) * sc[idx - (k - 1) * S[c]];
      else if (k == 1 && am == 1 && clamped_at<DEEP>(clamped, c, 0))
        acc -= 2.f * sc[idx];
    }
    force += F[c] * acc;
  }
#if NDIM >= 2
#pragma unroll
  for (int l = 0; l < NDIM; ++l) { // shear stresses along the other axes
    if (l == c)
      continue;
    const int v = PAIR_ROW(min(c, l), max(c, l));
    const real_t *__restrict__ sv = sigma + v * cs;
    real_t acc = (real_t)0;
#pragma unroll
    for (int j = 1; j <= STENCIL_RADIUS; ++j) {
      const int ap = A[l] + j - 1;
      const int rp =
          (DEEP || ap <= NN[l] - 3) ? half_radius<DEEP>(ap, NN[l]) : 0;
      if (j <= rp)
        acc += SG_W(rp, j) * sv[idx + (j - 1) * S[l]];
      const int am = A[l] - j;
      const int rm = (DEEP || am >= 1) ? half_radius<DEEP>(am, NN[l]) : 0;
      if (j <= rm)
        acc -= SG_W(rm, j) * sv[idx - j * S[l]];
    }
    force += F[l] * acc;
  }
#endif
  return force;
}

// ------------------------------------- pml helpers
// the axes a traction wall zeroes at a node: normal_strains' set bits alone
__device__ __forceinline__ int zeroed_axes(const int *A, const int *NN,
                                           const int clamped) {
  int zeroed = 0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    if (rad_node(A[d], NN[d]) == 0 &&
        !(A[d] == 1 && clamped_face(clamped, d, 0)) &&
        !(A[d] == NN[d] - 2 && clamped_face(clamped, d, 1)))
      zeroed |= 1 << d;
  return zeroed;
}

// the stress row (c, e): normal on the diagonal, else its shear pair
__device__ __forceinline__ int stress_row(const int c, const int e) {
  return c == e ? c : PAIR_ROW(min(c, e), max(c, e));
}

// stress (c, e) j nodes along axis e from a node, what component c diverges
struct Stress {
  const real_t *__restrict__ p;
  int idx, cs, step, e;
  __device__ real_t operator()(const int c, const int j) const {
    return p[stress_row(c, e) * cs + idx + j * step];
  }
};

// d u_k / d x_e at the strain point (k, e), graded at the walls as the step
template <class F>
__device__ __forceinline__ real_t strain_term(const F &u, const int k,
                                              const int e, const int *A,
                                              const int *NN, const real_t fe,
                                              const int clamped) {
  real_t acc = (real_t)0;
  if (k == e) {
    const int r = rad_node(A[e], NN[e]);
    if (r > 0) {
#pragma unroll
      for (int q = 1; q <= STENCIL_RADIUS; ++q)
        if (q <= r)
          acc += SG_W(r, q) * (u(e, q - 1) - u(e, -q));
    } else if (A[e] == 1 && clamped_face(clamped, e, 0)) {
      acc = 2.f * u(e, 0);
    } else if (A[e] == NN[e] - 2 && clamped_face(clamped, e, 1)) {
      acc = -2.f * u(e, -1);
    }
    return fe * acc;
  }
  if (A[k] > NN[k] - 3 || A[e] > NN[e] - 3)
    return (real_t)0; // no shear point there
  const int r = rad_half(A[e], NN[e]);
#pragma unroll
  for (int j = 1; j <= STENCIL_RADIUS; ++j)
    if (j <= r)
      acc += SG_W(r, j) * (u(k, j) - u(k, -(j - 1)));
  return fe * acc;
}

// the axis-e part of the divergence of component c, as fd_kernel takes it
template <class F>
__device__ __forceinline__ real_t divergence_term(const F &sig, const int c,
                                                  const int e, const int *A,
                                                  const int *NN,
                                                  const real_t fe,
                                                  const int clamped) {
  real_t acc = (real_t)0;
  if (c == e) {
#pragma unroll
    for (int k = 1; k <= STENCIL_RADIUS; ++k) {
      const int ap = A[c] + k, rp = rad_node(ap, NN[c]);
      if (k <= rp)
        acc += SG_W(rp, k) * sig(c, k);
      else if (k == 1 && ap == NN[c] - 2 && clamped_face(clamped, c, 1))
        acc += 2.f * sig(c, 1);
      const int am = A[c] - k + 1, rm = (am >= 1) ? rad_node(am, NN[c]) : 0;
      if (k <= rm)
        acc -= SG_W(rm, k) * sig(c, -(k - 1));
      else if (k == 1 && am == 1 && clamped_face(clamped, c, 0))
        acc -= 2.f * sig(c, 0);
    }
    return fe * acc;
  }
#pragma unroll
  for (int j = 1; j <= STENCIL_RADIUS; ++j) {
    const int ap = A[e] + j - 1;
    const int rp = (ap <= NN[e] - 3) ? rad_half(ap, NN[e]) : 0;
    if (j <= rp)
      acc += SG_W(rp, j) * sig(c, j - 1);
    const int am = A[e] - j, rm = (am >= 1) ? rad_half(am, NN[e]) : 0;
    if (j <= rm)
      acc -= SG_W(rm, j) * sig(c, -j);
  }
  return fe * acc;
}

// G times the strain terms t of axis e, added onto the stress points
__device__ __forceinline__ void
add_stress(real_t *__restrict__ sigma, const real_t *__restrict__ gamma,
           const int idx, const int cs, const int e, const real_t *t,
           const int *A, const int *S, const int *NN, const real_t lam,
           const real_t mu, const int clamped) {
  const int zeroed = zeroed_axes(A, NN, clamped);
  if (!((zeroed >> e) & 1)) {
    const real_t lam_eff = condensed_lame(lam, mu, zeroed);
    const real_t g = node_gamma(gamma, idx, A, NN) * t[e];
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      if (!((zeroed >> d) & 1))
        sigma[d * cs + idx] += g * ((d == e ? 2.f * mu : (real_t)0) + lam_eff);
  }
#if NDIM >= 2
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
    if (k != e && A[k] <= NN[k] - 3 && A[e] <= NN[e] - 3) {
      const int v = stress_row(k, e);
      sigma[v * cs + idx] +=
          shear_gamma(gamma, idx, min(k, e), max(k, e), A, S, NN) * mu * t[k];
    }
#endif
}

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
stress_kernel(const real_t *__restrict__ u1, real_t *__restrict__ sigma,
              const real_t *__restrict__ gamma, const real_t lam,
              const real_t mu, const int clamped, const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

  real_t sig[NVOIGT];
  if (deep_node(A, NN))
    stress_point<true>(u1, gamma, idx, cs, A, S, NN, F, lam, mu, clamped, sig);
  else
    stress_point(u1, gamma, idx, cs, A, S, NN, F, lam, mu, clamped, sig);
#pragma unroll
  for (int v = 0; v < NVOIGT; ++v)
    sigma[v * cs + idx] = sig[v];
}

// ------------------------------------------------------------------------------------
__global__ void
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ sigma,
          const real_t *__restrict__ gamma, const real_t rho,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
          const int clamped, const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

  const bool deep = deep_node(A, NN);
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    const real_t force =
        deep ? nodal_force<true>(sigma, idx, cs, S, c, A, NN, F, clamped)
             : nodal_force(sigma, idx, cs, S, c, A, NN, F, clamped);
    const int n = c * cs + idx;
    const real_t mi =
        deep ? point_minv<true>(gamma, idx, c, A, S, NN, rho, clamped)
             : point_minv(gamma, idx, c, A, S, NN, rho, clamped);
#ifdef USE_DAMPING
    const real_t beta = 0.5f * mi * damping[idx] * dt;
    u2[n] = (2.f * u1[n] - u0[n] * (1.f - beta) + mi * force) / (1.f + beta);
#else
    u2[n] = -u0[n] + 2.f * u1[n] + mi * force;
#endif
  }
}

// ------------------------------------------------------------------------------------
__global__ void
pml_stress_kernel(const real_t *__restrict__ u1, real_t *__restrict__ sigma,
                  const real_t *__restrict__ gamma, const real_t lam,
                  const real_t mu, const int clamped,
                  const int cs, real_t *__restrict__ aux,
                  const real_t *__restrict__ profile, const int e, const int lo,
                  const int hi, AXIS_PARAMS) {
  PML_AXES;
  SLAB_OR_RETURN(0);
  const Field u = {u1, idx, cs, S[e]};
  real_t psi[NDIM];
#pragma unroll
  for (int k = 0; k < NDIM; ++k) {
    const int at = (k == e ? 0 : 2 * NN[e]) + a; // node, or the half point
    psi[k] = profile[NN[e] + at] * aux[k * size + t] +
             profile[at] * strain_term(u, k, e, A, NN, F[e], clamped);
    aux[k * size + t] = psi[k];
  }
  add_stress(sigma, gamma, idx, cs, e, psi, A, S, NN, lam, mu, clamped);
}

// ------------------------------------------------------------------------------------
__global__ void
pml_force_kernel(const real_t *__restrict__ u1, real_t *__restrict__ u2,
                 const real_t *__restrict__ sigma,
                 const real_t *__restrict__ gamma, const real_t rho,
                 const int clamped, const int cs, real_t *__restrict__ aux,
                 const real_t *__restrict__ profile, const int e, const int lo,
                 const int hi, AXIS_PARAMS) {
  // u1 unread: the adjoint's signature, so one launch list serves both
  PML_AXES;
  SLAB_OR_RETURN(0);
  const Stress sig = {sigma, idx, cs, S[e], e};
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    const int at = (c == e ? 2 * NN[e] : 0) + a; // half point, or the node
    real_t *z = aux + (NDIM + c) * size + t;
    *z = profile[NN[e] + at] * *z +
         profile[at] * divergence_term(sig, c, e, A, NN, F[e], clamped);
    u2[c * cs + idx] += point_minv(gamma, idx, c, A, S, NN, rho, clamped) * *z;
  }
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_stress_kernel(
    const real_t *__restrict__ l1, real_t *__restrict__ sigma,
    const real_t *__restrict__ gamma, const real_t lam, const real_t mu,
    const int clamped, const int cs,
    const real_t *__restrict__ aux, const real_t *__restrict__ profile,
    const int e, const int lo, const int hi, AXIS_PARAMS) {
  PML_AXES;
  SLAB_OR_RETURN(STENCIL_RADIUS);
  // the divergence memory pushed back through the strain: the force's transpose
  const Memory<Field> h = {{l1, idx, cs, S[e]},
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
    q[k] = strain_term(h, k, e, A, NN, F[e], clamped);
  add_stress(sigma, gamma, idx, cs, e, q, A, S, NN, lam, mu, clamped);
}

// ------------------------------------------------------------------------------------
__global__ void
pml_adjoint_force_kernel(const real_t *__restrict__ l1, real_t *__restrict__ l2,
                         const real_t *__restrict__ sigma,
                         const real_t *__restrict__ gamma, const real_t rho,
                         const int clamped, const int cs,
                         real_t *__restrict__ aux,
                         const real_t *__restrict__ profile, const int e,
                         const int lo, const int hi, AXIS_PARAMS) {
  PML_AXES;
  SLAB_OR_RETURN(STENCIL_RADIUS);
  // the strain memory pushed back through the divergence: the stress' transpose
  const Memory<Stress> k = {{sigma, idx, cs, S[e], e},
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
    if (A[c] > NN[c] - 3)
      continue;
    const int n = c * cs + idx;
    l2[n] += point_minv(gamma, idx, c, A, S, NN, rho, clamped) *
             divergence_term(k, c, e, A, NN, F[e], clamped);
    if (t >= 0) { // the divergence memory, read by the stress pass before
      const int at = (c == e ? 2 * NN[e] : 0) + a;
      real_t *z = aux + (NDIM + c) * size + t;
      *z = profile[NN[e] + at] * (*z + l1[n]);
    }
  }
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_memory_kernel(const real_t *__restrict__ sigma,
                                          const int cs,
                                          real_t *__restrict__ aux,
                                          const real_t *__restrict__ profile,
                                          const int e, const int lo,
                                          const int hi, AXIS_PARAMS) {
  PML_AXES;
  SLAB_OR_RETURN(0);
#pragma unroll
  for (int k = 0; k < NDIM; ++k) {
    const int at = (k == e ? 0 : 2 * NN[e]) + a;
    real_t *m = aux + k * size + t;
    *m = profile[NN[e] + at] * (*m + sigma[stress_row(k, e) * cs + idx]);
  }
}

} // extern "C"
