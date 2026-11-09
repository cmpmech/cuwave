// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3
//   USE_DAMPING
//   USE_PML
//   RELAXATION = L (no pml, no TILE, no STEPS): L memory variables a stress
//   TILE0, TILE1, TILE2 (no pml): the fused step's tile
//   STEPS, TILE0, TILE1, ROWS (2D, no pml): the blocked step's
//   GRADIENT (no STEPS): the step adds the gradient densities of u1 against v1

#ifdef USE_PML
// the slowest axis' blocks alternately from either end, so the slabs there run
// in the first wave rather than trailing the last
#define BLOCK_ROW(b, n) ((b) % 2 ? (n) - 1 - (b) / 2 : (b) / 2)
#else
#define BLOCK_ROW(b, n) (b)
#endif

#if NDIM == 1
#define AXIS_PARAMS const real_t f0, const int N0
#define INTERIOR_OR_RETURN                                                     \
  const int a0 = BLOCK_ROW(blockIdx.x, gridDim.x) * blockDim.x + threadIdx.x;  \
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
  const int a0 = BLOCK_ROW(blockIdx.y, gridDim.y) * blockDim.y + threadIdx.y;  \
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
  const int a0 = BLOCK_ROW(blockIdx.z, gridDim.z) * blockDim.z + threadIdx.z;  \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&            \
        a2 < N2 - 1))                                                          \
    return;                                                                    \
  const int idx = a0 * s0 + a1 * s1 + a2
#define AXIS_GEOM                                                              \
  const int A[3] = {a0, a1, a2}, S[3] = {s0, s1, 1}, NN[3] = {N0, N1, N2}
#define AXIS_FACTORS const real_t F[3] = {f0, f1, f2}
#endif

// per axis the pml memory, its profile and its rows on either side, 0 unlined;
// phase 0 forwards, 1 or 2 the adjoint reading its strain memory off row 0 or
// 2 NDIM and settling it into the other
#ifdef USE_PML
#if NDIM == 1
#define PML_FOLD                                                               \
  , real_t *__restrict__ m0, const real_t *__restrict__ p0, const int lo0,     \
      const int hi0, const int phase
#define FOLD_GEOM                                                              \
  real_t *M[1] = {m0};                                                         \
  const real_t *P[1] = {p0};                                                   \
  const int LO[1] = {lo0}, HI[1] = {hi0}
#elif NDIM == 2
#define PML_FOLD                                                               \
  , real_t *__restrict__ m0, const real_t *__restrict__ p0, const int lo0,     \
      const int hi0, real_t *__restrict__ m1, const real_t *__restrict__ p1,   \
      const int lo1, const int hi1, const int phase
#define FOLD_GEOM                                                              \
  real_t *M[2] = {m0, m1};                                                     \
  const real_t *P[2] = {p0, p1};                                               \
  const int LO[2] = {lo0, lo1}, HI[2] = {hi0, hi1}
#elif NDIM == 3
#define PML_FOLD                                                               \
  , real_t *__restrict__ m0, const real_t *__restrict__ p0, const int lo0,     \
      const int hi0, real_t *__restrict__ m1, const real_t *__restrict__ p1,   \
      const int lo1, const int hi1, real_t *__restrict__ m2,                   \
      const real_t *__restrict__ p2, const int lo2, const int hi2,             \
      const int phase
#define FOLD_GEOM                                                              \
  real_t *M[3] = {m0, m1, m2};                                                 \
  const real_t *P[3] = {p0, p1, p2};                                           \
  const int LO[3] = {lo0, lo1, lo2}, HI[3] = {hi0, hi1, hi2}
#endif
#else
#define PML_FOLD
#endif

// the adjoint pair's accumulators and the forward field v1 they pair u1 with
#ifdef GRADIENT
#if NDIM >= 2
#define STIFFNESS_GRADIENT                                                     \
  , real_t *__restrict__ g_normal, real_t *__restrict__ g_shear,               \
      const real_t *__restrict__ v1
#define SHEAR_ROWS , g_shear
#else
#define STIFFNESS_GRADIENT                                                     \
  , real_t *__restrict__ g_normal, const real_t *__restrict__ v1
#define SHEAR_ROWS
#endif
#define MASS_GRADIENT                                                          \
  , real_t *__restrict__ g_mass, const real_t *__restrict__ v1, const real_t mf
#else
#define STIFFNESS_GRADIENT
#define MASS_GRADIENT
#endif

// the memory variables, RELAXATION rows per Voigt row, and their coefficients
#ifdef RELAXATION
#define RELAXATION_ARGS                                                        \
  , real_t *__restrict__ memory, const real_t *__restrict__ coef
#else
#define RELAXATION_ARGS
#endif

template <int DEEP>
__device__ __forceinline__ bool clamped_at(const int clamped, const int d,
                                           const int side) {
  return !((DEEP >> d) & 1) && clamped_face(clamped, d, side);
}

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
// d u_k / d x_l + d u_l / d x_k, rounded alike in every kernel and path
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

// whether the (k, l) shear point exists: off the staggered ghosts of k and l
template <int DEEP = 0>
__device__ __forceinline__ bool shear_point(const int k, const int l,
                                            const int *A, const int *NN) {
  return (((DEEP >> k) & 1) || A[k] <= NN[k] - 3) &&
         (((DEEP >> l) & 1) || A[l] <= NN[l] - 3);
}

// ---------------------------------- material helpers
#ifdef USE_FLOAT
#define TINY 1.17549435e-38f // finfo.tiny, the guard the python averages take
#else
#define TINY 2.2250738585072014e-308
#endif

#if NDIM >= 2
// the design on the (k, l) shear point: harmonic over the four nodes around it
__device__ __forceinline__ real_t shear_gamma(const real_t *__restrict__ gamma,
                                              const int idx, const int k,
                                              const int l, const int *S) {
  real_t sum = (real_t)1 / fmax(gamma[idx], TINY);
  sum += (real_t)1 / fmax(gamma[idx + S[l]], TINY);
  sum += (real_t)1 / fmax(gamma[idx + S[k]], TINY);
  sum += (real_t)1 / fmax(gamma[idx + S[k] + S[l]], TINY);
  return (real_t)4 / sum;
}
#endif

// inverse inertia of component c's point, zero where a clamped wall holds it
template <int DEEP = 0>
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
  return (real_t)1 / fmax(rho * mean, TINY);
}

// ----------------------------------- step helpers
// every strain of the node at idx, zero off its point, and the axes a traction
// wall zeroes
template <int DEEP = 0>
__device__ __forceinline__ int
strain_point(const real_t *__restrict__ u, const int idx, const int cs,
             const int *A, const int *S, const int *NN, const real_t *F,
             const int clamped, real_t *eps) {
  const int zeroed =
      normal_strains<DEEP>(u, idx, cs, A, S, NN, F, clamped, eps);
#if NDIM >= 2
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l)
      eps[PAIR_ROW(k, l)] = shear_point<DEEP>(k, l, A, NN)
                                ? shear_strain<DEEP>(u, idx, cs, k, l, A, S,
                                                     NN, F)
                                : (real_t)0;
#endif
  return zeroed;
}

// the stresses of the strains eps at the node at idx, zero off their points
template <int DEEP = 0>
__device__ __forceinline__ void
stress_of(const real_t *eps, const int zeroed,
          const real_t *__restrict__ gamma, const int idx, const int *A,
          const int *S, const int *NN, const real_t lam, const real_t mu,
          real_t *sig) {
  const real_t lam_eff = condensed_lame(lam, mu, zeroed);
  real_t tr = (real_t)0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    tr += eps[d];
  const real_t gn = gamma[idx];
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
      sig[v] = shear_point<DEEP>(k, l, A, NN)
                   ? shear_gamma(gamma, idx, k, l, S) * mu * eps[v]
                   : (real_t)0;
    }
#endif
}

// every stress of the node at idx, zero off its point
template <int DEEP = 0>
__device__ __forceinline__ void
stress_point(const real_t *__restrict__ u, const real_t *__restrict__ gamma,
             const int idx, const int cs, const int *A, const int *S,
             const int *NN, const real_t *F, const real_t lam, const real_t mu,
             const int clamped, real_t *sig) {
  real_t eps[NVOIGT];
  const int zeroed =
      strain_point<DEEP>(u, idx, cs, A, S, NN, F, clamped, eps);
  stress_of<DEEP>(eps, zeroed, gamma, idx, A, S, NN, lam, mu, sig);
}

// the stress divergence on component c's point, off the accessor (sigma, cs, S)
template <int DEEP = 0>
__device__ __forceinline__ real_t
nodal_force(const real_t *__restrict__ sigma, const int idx, const int cs,
            const int *S, const int c, const int *A, const int *NN,
            const real_t *F, const int clamped) {
  // own normal stress along axis c, then the shear stresses along the others,
  // rounded alike in every kernel
  real_t force =
      MUL_RN(F[c], node_divergence(Line{sigma + c * cs, idx, S[c]}, A[c], NN[c],
                                   (DEEP >> c) & 1));
#if NDIM >= 2
#pragma unroll
  for (int l = 0; l < NDIM; ++l) {
    if (l == c)
      continue;
    const int v = PAIR_ROW(min(c, l), max(c, l));
    force = FMA_RN(F[l],
                   half_divergence(Line{sigma + v * cs, idx, S[l]}, A[l], NN[l],
                                   (DEEP >> l) & 1),
                   force);
  }
#endif
  return force;
}

#ifdef RELAXATION
// the node's Voigt rows s relaxed by their memory variables, each settled in
// place: s <- c0 s - sum read_l m_l, then m_l <- a_l m_l + drive_l s
__device__ __forceinline__ void relax(real_t *__restrict__ memory,
                                      const real_t *__restrict__ coef,
                                      const int idx, const int cs, real_t *s) {
#pragma unroll
  for (int v = 0; v < NVOIGT; ++v) {
    real_t out = coef[0] * s[v];
#pragma unroll
    for (int l = 0; l < RELAXATION; ++l) {
      real_t *__restrict__ m = memory + (l * NVOIGT + v) * cs + idx;
      const real_t old = *m;
      out -= coef[3 * l + 2] * old;
      *m = coef[3 * l + 1] * old + coef[3 * l + 3] * s[v];
    }
    s[v] = out;
  }
}
#endif

#ifdef GRADIENT
// ------------------------------------ gradient helpers
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

// dJ/dgamma on the node and its shear points: the adjoint's strains eps, under
// a relaxation relaxed, against the forward field v1's
template <int DEEP = 0>
__device__ __forceinline__ void
stiffness_density(const real_t *eps, const real_t *__restrict__ v1,
                  real_t *__restrict__ g_normal,
#if NDIM >= 2
                  real_t *__restrict__ g_shear,
#endif
                  const int idx, const int cs, const int *A, const int *S,
                  const int *NN, const real_t *F, const real_t lam,
                  const real_t mu, const int clamped) {
  real_t eu[NDIM];
  const int zeroed =
      normal_strains<DEEP>(v1, idx, cs, A, S, NN, F, clamped, eu);
  g_normal[idx] -= normal_form(eps, eu, lam, mu, zeroed);

#if NDIM >= 2
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      if (!shear_point<DEEP>(k, l, A, NN))
        continue;
      const int v = PAIR_ROW(k, l);
      g_shear[(v - NDIM) * cs + idx] -=
          mu * eps[v] * shear_strain<DEEP>(v1, idx, cs, k, l, A, S, NN, F);
    }
#endif
}
#endif

// ------------------------------------- pml helpers
// the axes a traction wall zeroes at a node: normal_strains' set bits alone
__device__ __forceinline__ int zeroed_axes(const int *A, const int *NN,
                                           const int clamped) {
  int zeroed = 0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    if ((A[d] == 1 && !clamped_face(clamped, d, 0)) ||
        (A[d] == NN[d] - 2 && !clamped_face(clamped, d, 1)))
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

// component k of an accessor of (k, j), a line along the accessor's axis
template <class F> struct Along {
  F f;
  int k;
  __device__ real_t operator()(const int j) const { return f(k, j); }
};

// d u_k / d x_e at the strain point (k, e), the step's closure rows
template <class F>
__device__ __forceinline__ real_t strain_term(const F &u, const int k,
                                              const int e, const int *A,
                                              const int *NN, const real_t fe,
                                              const int clamped) {
  if (k == e) {
    if ((A[e] == 1 && !clamped_face(clamped, e, 0)) ||
        (A[e] == NN[e] - 2 && !clamped_face(clamped, e, 1)))
      return (real_t)0; // a traction wall's zero row
    return fe * node_derivative(Along<F>{u, e}, A[e], NN[e]);
  }
  if (A[k] > NN[k] - 3 || A[e] > NN[e] - 3)
    return (real_t)0; // no shear point there
  return fe * half_derivative(Along<F>{u, k}, A[e], NN[e]);
}

// the axis-e part of the divergence of component c, as fd_kernel takes it
template <class F>
__device__ __forceinline__ real_t divergence_term(const F &sig, const int c,
                                                  const int e, const int *A,
                                                  const int *NN,
                                                  const real_t fe) {
  if (c == e)
    return fe * node_divergence(Along<F>{sig, c}, A[c], NN[c]);
  return fe * half_divergence(Along<F>{sig, c}, A[e], NN[e]);
}

// G times the strain terms t of axis e, added onto the node's stresses sig
__device__ __forceinline__ void
add_stress(real_t *sig, const real_t *__restrict__ gamma, const int idx,
           const int e, const real_t *t, const int *A, const int *S,
           const int *NN, const real_t lam, const real_t mu,
           const int clamped) {
  const int zeroed = zeroed_axes(A, NN, clamped);
  if (!((zeroed >> e) & 1)) {
    const real_t lam_eff = condensed_lame(lam, mu, zeroed);
    const real_t g = gamma[idx] * t[e];
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      if (!((zeroed >> d) & 1))
        sig[d] += g * ((d == e ? 2.f * mu : (real_t)0) + lam_eff);
  }
#if NDIM >= 2
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
    if (k != e && A[k] <= NN[k] - 3 && A[e] <= NN[e] - 3) {
      sig[stress_row(k, e)] +=
          shear_gamma(gamma, idx, min(k, e), max(k, e), S) * mu * t[k];
    }
#endif
}

// the slab offset of a node off axis d: the other axes flattened, as slab_node
__device__ __forceinline__ int slab_offset(const int d, const int *A,
                                           const int *NN, const int *S) {
  int off = 0;
#pragma unroll
  for (int e = 0; e < NDIM; ++e)
    if (e != d)
      off = off * padded_extent(e, NN, S) + A[e];
  return off;
}

// whether a node lies in axis e's slab, grown ext rows past its front
__device__ __forceinline__ bool in_slab(const int a, const int e, const int lo,
                                        const int hi, const int ext,
                                        const int *NN) {
  return (lo && a - 1 < lo + ext) || (hi && NN[e] - 2 - a < hi + ext);
}

// axis e's memory terms on the node's stresses: the strain memory, or in the
// adjoint the divergence memory pushed back through the strain
__device__ __forceinline__ void
stress_memory(const real_t *__restrict__ u, const real_t *__restrict__ gamma,
              const int idx, const int cs, const int e,
              real_t *__restrict__ aux, const real_t *__restrict__ profile,
              const int lo, const int hi, const int phase, const int *A,
              const int *S, const int *NN, const real_t *F, const real_t lam,
              const real_t mu, const int clamped, real_t *sig) {
  const int a = A[e];
  if (!in_slab(a, e, lo, hi, phase ? STENCIL_RADIUS : 0, NN))
    return;
  const int plane = slab_plane(e, NN, S), size = (lo + hi) * plane;
  const int off = slab_offset(e, A, NN, S);
  real_t q[NDIM];
  if (phase) {
    const Memory<Field> h = {{u, idx, cs, S[e]},
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
#pragma unroll
    for (int k = 0; k < NDIM; ++k)
      q[k] = strain_term(h, k, e, A, NN, F[e], clamped);
  } else {
    const Field f = {u, idx, cs, S[e]};
    const int t = slab_index(a, off, e, lo, hi, NN, plane);
#pragma unroll
    for (int k = 0; k < NDIM; ++k) {
      const int at = (k == e ? 0 : 2 * NN[e]) + a; // node, or the half point
      q[k] = profile[NN[e] + at] * aux[k * size + t] +
             profile[at] * strain_term(f, k, e, A, NN, F[e], clamped);
      aux[k * size + t] = q[k];
    }
  }
  add_stress(sig, gamma, idx, e, q, A, S, NN, lam, mu, clamped);
}

// axis e's memory term on component c's force: the divergence memory, or in
// the adjoint the strain memory pushed back, its own memory then taking l1
__device__ __forceinline__ real_t
force_memory(const real_t *__restrict__ sigma, const real_t *__restrict__ u1,
             const int idx, const int cs, const int c, const int e,
             real_t *__restrict__ aux, const real_t *__restrict__ profile,
             const int lo, const int hi, const int phase, const int *A,
             const int *S, const int *NN, const real_t *F) {
  const int a = A[e];
  if (!in_slab(a, e, lo, hi, phase ? STENCIL_RADIUS : 0, NN))
    return (real_t)0;
  const int plane = slab_plane(e, NN, S), size = (lo + hi) * plane;
  const int off = slab_offset(e, A, NN, S);
  const int t = slab_index(a, off, e, lo, hi, NN, plane);
  const int at = (c == e ? 2 * NN[e] : 0) + a; // half point, or the node
  real_t *z = aux + (NDIM + c) * size + t;
  if (phase) {
    const Memory<Stress> k = {{sigma, idx, cs, S[e], e},
                              aux,
                              profile,
                              NN,
                              size,
                              phase == 1 ? 0 : 2 * NDIM,
                              a,
                              off,
                              e,
                              lo,
                              hi,
                              plane,
                              false};
    const real_t term = divergence_term(k, c, e, A, NN, F[e]);
    if (t >= 0) // the divergence memory, read by the stress pass before
      *z = profile[NN[e] + at] * (*z + u1[c * cs + idx]);
    return term;
  }
  const Stress sig = {sigma, idx, cs, S[e], e};
  *z = profile[NN[e] + at] * *z +
       profile[at] * divergence_term(sig, c, e, A, NN, F[e]);
  return *z;
}

// the adjoint's strain memory of axis e, read off one row block and settled
// with this step's stress into the other, which the next step reads
__device__ __forceinline__ void
settle_memory(const real_t *__restrict__ sigma, const int idx, const int cs,
              const int e, real_t *__restrict__ aux,
              const real_t *__restrict__ profile, const int lo, const int hi,
              const int phase, const int *A, const int *S, const int *NN) {
  const int a = A[e];
  if (!in_slab(a, e, lo, hi, 0, NN))
    return;
  const int plane = slab_plane(e, NN, S), size = (lo + hi) * plane;
  const int t = slab_index(a, slab_offset(e, A, NN, S), e, lo, hi, NN, plane);
  const int from = phase == 1 ? 0 : 2 * NDIM, to = 2 * NDIM - from;
#pragma unroll
  for (int k = 0; k < NDIM; ++k) {
    const int at = (k == e ? 0 : 2 * NN[e]) + a;
    aux[(to + k) * size + t] =
        profile[NN[e] + at] *
        (aux[(from + k) * size + t] + sigma[stress_row(k, e) * cs + idx]);
  }
}

// --------------------------------- fused step helpers
#if NDIM >= 2 && defined(TILE0) && !defined(USE_PML) && !defined(STEPS)
// a thread a tile node, the stresses on the tile and a ring a radius wide
#define WIDE (TILE1 + 2 * STENCIL_RADIUS)
#if NDIM == 2
#define RING ((TILE0 + 2 * STENCIL_RADIUS) * WIDE)
#define THREADS (TILE0 * TILE1)
#define TILE_INTERIOR_OR_RETURN                                                \
  const int a0 = o0 + threadIdx.y, a1 = o1 + threadIdx.x;                      \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 &&                      \
        a0 < tile_end(blockIdx.y, TILE0, N0) &&                                \
        a1 < tile_end(blockIdx.x, TILE1, N1)))                                 \
    return;                                                                    \
  const int idx = a0 * s0 + a1
#else
#define DEEP_ (TILE2 + 2 * STENCIL_RADIUS)
#define RING ((TILE0 + 2 * STENCIL_RADIUS) * WIDE * DEEP_)
#define THREADS (TILE0 * TILE1 * TILE2)
#define TILE_INTERIOR_OR_RETURN                                                \
  const int a0 = o0 + threadIdx.z, a1 = o1 + threadIdx.y;                      \
  const int a2 = o2 + threadIdx.x;                                             \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&            \
        a2 < N2 - 1 && a0 < tile_end(blockIdx.z, TILE0, N0) &&                 \
        a1 < tile_end(blockIdx.y, TILE1, N1) &&                                \
        a2 < tile_end(blockIdx.x, TILE2, N2)))                                 \
    return;                                                                    \
  const int idx = a0 * s0 + a1 * s1 + a2
#endif

// a ring point of a band warp: its stresses, deep along the axes deep, into out
// a shared plane apart; out of line, so the deep ring keeps few registers
#if NDIM == 2
__device__ __noinline__ void
wall_stresses(const real_t *__restrict__ u1, const real_t *__restrict__ gamma,
              const int cs, const real_t lam, const real_t mu,
              const int clamped, const int deep, const int g0, const int g1,
              const int N0, const int N1, const int s0, const real_t e0,
              const real_t e1, real_t *out) {
  const int G[2] = {g0, g1}, GS[2] = {s0, 1}, GN[2] = {N0, N1};
  const real_t E[2] = {e0, e1};
  const int at = g0 * s0 + g1;
#else
__device__ __noinline__ void
wall_stresses(const real_t *__restrict__ u1, const real_t *__restrict__ gamma,
              const int cs, const real_t lam, const real_t mu,
              const int clamped, const int deep, const int g0, const int g1,
              const int g2, const int N0, const int N1, const int N2,
              const int s0, const int s1, const real_t e0, const real_t e1,
              const real_t e2, real_t *out) {
  const int G[3] = {g0, g1, g2}, GS[3] = {s0, s1, 1}, GN[3] = {N0, N1, N2};
  const real_t E[3] = {e0, e1, e2};
  const int at = g0 * s0 + g1 * s1 + g2;
#endif
  real_t sig[NVOIGT];
  if (deep == DEEP_SLOW)
    stress_point<DEEP_SLOW>(u1, gamma, at, cs, G, GS, GN, E, lam, mu, clamped,
                            sig);
  else
    stress_point(u1, gamma, at, cs, G, GS, GN, E, lam, mu, clamped, sig);
#pragma unroll
  for (int v = 0; v < NVOIGT; ++v)
    out[v * RING] = sig[v];
}

// the stresses of the tile's ring, whose first node is B, into ss: those of
// stress_kernel's nodes, zero elsewhere; DEEP where all of them are deep
template <bool DEEP>
__device__ __forceinline__ void
ring_stresses(real_t *ss, const real_t *__restrict__ u1,
              const real_t *__restrict__ gamma, const int cs, const real_t lam,
              const real_t mu, const int clamped, const int *B, const int *GS,
              const int *GN, const real_t *E) {
#if NDIM == 2
  for (int t = threadIdx.y * TILE1 + threadIdx.x; t < RING; t += THREADS) {
    const int G[2] = {B[0] + t / WIDE, B[1] + t % WIDE};
#else
  for (int t = (threadIdx.z * TILE1 + threadIdx.y) * TILE2 + threadIdx.x;
       t < RING; t += THREADS) {
    const int G[3] = {B[0] + t / (WIDE * DEEP_), B[1] + t / DEEP_ % WIDE,
                      B[2] + t % DEEP_};
#endif
    bool inner = true;
    int at = 0;
#pragma unroll
    for (int d = 0; d < NDIM; ++d) {
      inner &= G[d] > 0 && G[d] < GN[d] - 1;
      at += G[d] * GS[d];
    }
    const int deep = DEEP || !inner ? DEEP_ALL : warp_deep(G, GN);
    if (deep != DEEP_ALL) {
#if NDIM == 2
      wall_stresses(u1, gamma, cs, lam, mu, clamped, deep, G[0], G[1], GN[0],
                    GN[1], GS[0], E[0], E[1], ss + t);
#else
      wall_stresses(u1, gamma, cs, lam, mu, clamped, deep, G[0], G[1], G[2],
                    GN[0], GN[1], GN[2], GS[0], GS[1], E[0], E[1], E[2],
                    ss + t);
#endif
      continue;
    }
    real_t sig[NVOIGT];
    if (DEEP || inner) {
      stress_point<DEEP_ALL>(u1, gamma, at, cs, G, GS, GN, E, lam, mu, clamped,
                             sig);
    } else {
#pragma unroll
      for (int v = 0; v < NVOIGT; ++v)
        sig[v] = (real_t)0;
    }
#pragma unroll
    for (int v = 0; v < NVOIGT; ++v)
      ss[v * RING + t] = sig[v];
  }
}

// whether a ring spanning [b, b + T + 2 R) of an axis of N nodes is deep
__device__ __forceinline__ bool deep_ring(const int b, const int T,
                                          const int N) {
  return b > CLOSURE_BAND &&
         b + T + 2 * STENCIL_RADIUS - 1 < N - 2 - CLOSURE_BAND;
}

// the first node of tile b of T along an axis of N nodes, -1 past the last: the
// last ends on the axis' end, so no wall's closure band straddles two tiles
__device__ __forceinline__ int tile_origin(const int b, const int T,
                                           const int N) {
  return b * T < N ? min(b * T, max(N - T, 0)) : -1;
}

// the end of the nodes tile b steps, the next tile's origin
__device__ __forceinline__ int tile_end(const int b, const int T, const int N) {
  const int next = tile_origin(b + 1, T, N);
  return next < 0 ? N : next;
}
#endif

// register caps keeping the deep path's occupancy (__maxnreg__ from CUDA 12.4)
#define REGISTER_CAPS                                                          \
  (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ >= 1204)
#if REGISTER_CAPS && NDIM == 2
#define STRESS_REGISTERS __maxnreg__(40)
#define UPDATE_REGISTERS __maxnreg__(40)
#elif REGISTER_CAPS && NDIM == 3 && defined(GRADIENT)
#define STRESS_REGISTERS __maxnreg__(64)
#define UPDATE_REGISTERS
#else
#define STRESS_REGISTERS
#define UPDATE_REGISTERS
#endif

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void STRESS_REGISTERS
stress_kernel(const real_t *__restrict__ u1, real_t *__restrict__ sigma,
              const real_t *__restrict__ gamma, const real_t lam,
              const real_t mu, const int clamped, const int cs,
              AXIS_PARAMS STIFFNESS_GRADIENT PML_FOLD RELAXATION_ARGS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

  const int deep = warp_deep(A, NN);
  real_t sig[NVOIGT];
#ifdef GRADIENT
  // the adjoint's strains once, for its density and its stresses, a relaxation
  // holding its memory in strains so the density pairs the relaxed ones
  real_t eps[NVOIGT];
  const int zeroed =
      BY_DEEP(deep, strain_point, u1, idx, cs, A, S, NN, F, clamped, eps);
#ifdef RELAXATION
  relax(memory, coef, idx, cs, eps);
#endif
  BY_DEEP(deep, stiffness_density, eps, v1, g_normal SHEAR_ROWS, idx, cs, A, S,
          NN, F, lam, mu, clamped);
  BY_DEEP(deep, stress_of, eps, zeroed, gamma, idx, A, S, NN, lam, mu, sig);
#else
  BY_DEEP(deep, stress_point, u1, gamma, idx, cs, A, S, NN, F, lam, mu, clamped,
          sig);
#ifdef RELAXATION
  relax(memory, coef, idx, cs, sig);
#endif
#endif
#ifdef USE_PML
  FOLD_GEOM;
#pragma unroll
  for (int e = 0; e < NDIM; ++e)
    if (LO[e] + HI[e])
      stress_memory(u1, gamma, idx, cs, e, M[e], P[e], LO[e], HI[e], phase, A,
                    S, NN, F, lam, mu, clamped, sig);
#endif
#pragma unroll
  for (int v = 0; v < NVOIGT; ++v)
    sigma[v * cs + idx] = sig[v];
}

// ------------------------------------------------------------------------------------
__global__ void UPDATE_REGISTERS
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ sigma,
          const real_t *__restrict__ gamma, const real_t rho,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
          const int clamped, const int cs, AXIS_PARAMS MASS_GRADIENT PML_FOLD) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;
#ifdef USE_PML
  FOLD_GEOM;
#endif

  const int deep = warp_deep(A, NN);
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    const real_t force =
        BY_DEEP(deep, nodal_force, sigma, idx, cs, S, c, A, NN, F, clamped);
    const int n = c * cs + idx;
    const real_t mi =
        BY_DEEP(deep, point_minv, gamma, idx, c, A, S, NN, rho, clamped);
#ifdef USE_DAMPING
    const real_t beta = 0.5f * mi * damping[idx] * dt;
    real_t next =
        (2.f * u1[n] - u0[n] * (1.f - beta) + mi * force) / (1.f + beta);
#else
    real_t next = -u0[n] + 2.f * u1[n] + mi * force;
#endif
#ifdef USE_PML
#pragma unroll
    for (int e = 0; e < NDIM; ++e)
      if (LO[e] + HI[e])
        next += mi * force_memory(sigma, u1, idx, cs, c, e, M[e], P[e], LO[e],
                                  HI[e], phase, A, S, NN, F);
#endif
#ifdef GRADIENT
    // dJ/dmass by parts in time: v against the second difference of lambda
    g_mass[n] -= mf * v1[n] * (next - 2.f * u1[n] + u0[n]);
#endif
    u2[n] = next;
  }
#ifdef USE_PML
  if (phase) {
#pragma unroll
    for (int e = 0; e < NDIM; ++e)
      if (LO[e] + HI[e])
        settle_memory(sigma, idx, cs, e, M[e], P[e], LO[e], HI[e], phase, A, S,
                      NN);
  }
#endif
}

#if NDIM >= 2 && defined(TILE0) && !defined(USE_PML) && !defined(STEPS)
// ------------------------------------------------------------------------------------
// the stresses of the block's tile and a ring a stencil radius wide around it,
// then the update; at most 42 registers to radius 2, 64 past it
__global__ void
__launch_bounds__(THREADS, (STENCIL_RADIUS > 2 ? 1024 : 1536) / THREADS)
    fused_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
                 real_t *__restrict__ u2, const real_t *__restrict__ gamma,
                 const real_t lam, const real_t mu, const real_t rho,
#ifdef USE_DAMPING
                 const real_t *__restrict__ damping, const real_t dt,
#endif
                 const int clamped, const int cs, const real_t e0,
                 const real_t e1,
#if NDIM == 3
                 const real_t e2,
#endif
                 AXIS_PARAMS
#ifdef GRADIENT
                 ,
                 real_t *__restrict__ g_mass, real_t *__restrict__ g_normal,
                 real_t *__restrict__ g_shear, const real_t *__restrict__ v1,
                 const real_t mf
#endif
    ) {
  extern __shared__ real_t ss[]; // NVOIGT * RING
#if NDIM == 2
  const int o0 = tile_origin(blockIdx.y, TILE0, N0);
  const int o1 = tile_origin(blockIdx.x, TILE1, N1);
  if (o0 < 0 || o1 < 0)
    return;
  const int B[2] = {o0 - STENCIL_RADIUS, o1 - STENCIL_RADIUS};
  const int GS[2] = {s0, 1}, GN[2] = {N0, N1};
  const real_t E[2] = {e0, e1};
  const bool deep_tile =
      deep_ring(B[0], TILE0, N0) && deep_ring(B[1], TILE1, N1);
#else
  const int o0 = tile_origin(blockIdx.z, TILE0, N0);
  const int o1 = tile_origin(blockIdx.y, TILE1, N1);
  const int o2 = tile_origin(blockIdx.x, TILE2, N2);
  if (o0 < 0 || o1 < 0 || o2 < 0)
    return;
  const int B[3] = {o0 - STENCIL_RADIUS, o1 - STENCIL_RADIUS,
                    o2 - STENCIL_RADIUS};
  const int GS[3] = {s0, s1, 1}, GN[3] = {N0, N1, N2};
  const real_t E[3] = {e0, e1, e2};
  const bool deep_tile = deep_ring(B[0], TILE0, N0) &&
                         deep_ring(B[1], TILE1, N1) &&
                         deep_ring(B[2], TILE2, N2);
#endif
  if (deep_tile)
    ring_stresses<true>(ss, u1, gamma, cs, lam, mu, clamped, B, GS, GN, E);
  else
    ring_stresses<false>(ss, u1, gamma, cs, lam, mu, clamped, B, GS, GN, E);
  __syncthreads();

  // fd_kernel's update, the divergence read off the tile
  TILE_INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;
#if NDIM == 2
  const int local = (threadIdx.y + STENCIL_RADIUS) * WIDE + threadIdx.x +
                    STENCIL_RADIUS;
  const int T[2] = {WIDE, 1};
#else
  const int local =
      ((threadIdx.z + STENCIL_RADIUS) * WIDE + threadIdx.y + STENCIL_RADIUS) *
          DEEP_ +
      threadIdx.x + STENCIL_RADIUS;
  const int T[3] = {WIDE * DEEP_, DEEP_, 1};
#endif
  const int deep = deep_tile ? DEEP_ALL : warp_deep(A, NN);
#ifdef GRADIENT
  real_t eps[NVOIGT];
  BY_DEEP(deep, strain_point, u1, idx, cs, A, S, NN, E, clamped, eps);
  BY_DEEP(deep, stiffness_density, eps, v1, g_normal, g_shear, idx, cs, A, S,
          NN, E, lam, mu, clamped);
#endif
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    const real_t force =
        BY_DEEP(deep, nodal_force, ss, local, RING, T, c, A, NN, F, clamped);
    const int n = c * cs + idx;
    const real_t mi =
        BY_DEEP(deep, point_minv, gamma, idx, c, A, S, NN, rho, clamped);
#ifdef USE_DAMPING
    const real_t beta = 0.5f * mi * damping[idx] * dt;
    const real_t next =
        (2.f * u1[n] - u0[n] * (1.f - beta) + mi * force) / (1.f + beta);
#else
    const real_t next = -u0[n] + 2.f * u1[n] + mi * force;
#endif
#ifdef GRADIENT
    g_mass[n] -= mf * v1[n] * (next - 2.f * u1[n] + u0[n]);
#endif
    u2[n] = next;
  }
}
#endif

#if NDIM == 2 && defined(STEPS) && !defined(USE_PML)
// ------------------------------------------------------------------------------------
// the core a tile writes back, its halo STEPS strain-divergence reaches wide
#define HALO (STEPS * (2 * STENCIL_RADIUS - 1))
#define C0 (TILE0 - 2 * HALO)
#define C1 (TILE1 - 2 * HALO)
#define NODES (TILE0 / ROWS) // a thread's consecutive nodes along axis 0
#define WIDE (TILE1 + 2 * STENCIL_RADIUS) // a tile row with its guard columns
#define AREA ((TILE0 + 2 * STENCIL_RADIUS) * WIDE)
#define AT(x, y) (((x) + STENCIL_RADIUS) * WIDE + (y) + STENCIL_RADIUS)
static_assert(TILE0 % ROWS == 0, "the thread rows split the tile evenly");
static_assert(2 * NODES <= 32, "a component bit per node fits the mask");

// whether node a of a tile spanning [f, f + T) of an axis of N nodes reads only
// what the tile holds: off both walls' closure bands, or holding that wall
__device__ __forceinline__ bool reads_held(const int a, const int f,
                                           const int T, const int N) {
  return (a > CLOSURE_BAND || f <= 1) &&
         (a < N - 2 - CLOSURE_BAND || f + T - 1 >= N - 2);
}

// advance a tile STEPS steps between one load and one store: the stresses, the
// update, the excitation and the record, bitwise the plain launches
__global__ void __launch_bounds__(TILE1 *ROWS)
    blocked_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
                   real_t *__restrict__ v0, real_t *__restrict__ v1,
                   const real_t *__restrict__ gamma, const real_t lam,
                   const real_t mu, const real_t rho,
#ifdef USE_DAMPING
                   const real_t *__restrict__ damping, const real_t dt,
#endif
                   const real_t *__restrict__ signal,
                   const real_t *__restrict__ weights, const int num_sources,
                   real_t *__restrict__ record, const int num_sensors,
                   const int *__restrict__ source_starts,
                   const unsigned *__restrict__ source_entries,
                   const int *__restrict__ sensor_starts,
                   const unsigned *__restrict__ sensor_entries, const int t0,
                   const int clamped, const int cs, const real_t e0,
                   const real_t e1, const real_t f0, const int N0,
                   const real_t f1, const int N1, const int s0, const int o0,
                   const int o1) {
  // both components of u, the stresses and the design, each in a guard ring
  extern __shared__ real_t shared[];
  real_t *su = shared, *ss = shared + 2 * AREA, *sg = shared + 5 * AREA;
  const int ty = threadIdx.x, x0 = threadIdx.y * NODES;
  const int thread = threadIdx.y * TILE1 + ty;
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  const int lo0 = o0 + blockIdx.y * C0, lo1 = o1 + blockIdx.x * C1;
  const int a1 = lo1 - HALO + ty, k1 = owner(a1, N1);
  const bool col = a1 >= 0 && a1 <= N1 - 1;
  const bool core1 = k1 >= lo1 && k1 < lo1 + C1;
  const real_t E[2] = {e0, e1}, F[2] = {f0, f1};
  const int T[2] = {WIDE, 1}, GS[2] = {s0, 1}, NN[2] = {N0, N1};
  for (int e = thread; e < 6 * AREA; e += TILE1 * ROWS)
    shared[e] = 0;
  __syncthreads();

  // the thread's nodes, loaded once: u twice, 1 / m and beta per component
  real_t up[NODES][2], uc[NODES][2], mi[NODES][2];
#ifdef USE_DAMPING
  real_t beta[NODES][2];
#endif
  unsigned inner = 0, moving = 0; // bit k: an interior node; 2 k + c: c steps
#pragma unroll
  for (int k = 0; k < NODES; ++k) {
    const int x = x0 + k, a0 = lo0 - HALO + x, idx = a0 * s0 + a1;
    const bool inside = col && a0 >= 0 && a0 <= N0 - 1;
    const bool interior = a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 &&
                          reads_held(a0, lo0 - HALO, TILE0, N0) &&
                          reads_held(a1, lo1 - HALO, TILE1, N1);
    const int A[2] = {a0, a1};
    const int deep = warp_deep(A, NN);
    sg[AT(x, ty)] = inside ? gamma[idx] : (real_t)1;
    inner |= (unsigned)interior << k;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      up[k][c] = inside ? u0[c * cs + idx] : 0;
      uc[k][c] = inside ? u1[c * cs + idx] : 0;
      su[c * AREA + AT(x, ty)] = uc[k][c];
      const bool steps = interior && A[c] <= NN[c] - 3;
      moving |= (unsigned)steps << (2 * k + c);
      mi[k][c] = steps ? BY_DEEP(deep, point_minv, gamma, idx, c, A, GS, NN,
                                 rho, clamped)
                       : (real_t)0;
#ifdef USE_DAMPING
      beta[k][c] = steps ? 0.5f * mi[k][c] * damping[idx] : 0; // over dt
#endif
    }
  }
  __syncthreads();

  // the tile's sources and sensors as tags in the stress planes, source + 1 and
  // sensor + 1 a half, the component in the entry's local node
  unsigned *tags = (unsigned *)ss;
  for (int e = source_starts[tile] + thread; e < source_starts[tile + 1];
       e += TILE1 * ROWS) {
    const unsigned entry = source_entries[e], local = entry & 0xffff;
    const int c = local / (TILE0 * TILE1), at = local % (TILE0 * TILE1);
    atomicOr(&tags[c * AREA + AT(at / TILE1, at % TILE1)], (entry >> 16) + 1);
  }
  for (int e = sensor_starts[tile] + thread; e < sensor_starts[tile + 1];
       e += TILE1 * ROWS) {
    const unsigned entry = sensor_entries[e], local = entry & 0xffff;
    const int c = local / (TILE0 * TILE1), at = local % (TILE0 * TILE1);
    atomicOr(&tags[c * AREA + AT(at / TILE1, at % TILE1)],
             (entry >> 16) + 1 << 16);
  }
  __syncthreads();
  unsigned tag[NODES][2];
#pragma unroll
  for (int k = 0; k < NODES; ++k) {
    tag[k][0] = tags[AT(x0 + k, ty)];
    tag[k][1] = tags[AREA + AT(x0 + k, ty)];
  }
  __syncthreads();

  for (int t = 0; t < STEPS; ++t) {
    // stress_kernel's nodes, the tile edge forming garbage that never reaches
    // the core
#pragma unroll
    for (int k = 0; k < NODES; ++k) {
      const int x = x0 + k, a0 = lo0 - HALO + x;
      const int A[2] = {a0, a1};
      real_t sig[NVOIGT];
      if ((inner >> k) & 1) {
        BY_DEEP(warp_deep(A, NN), stress_point, su, sg, AT(x, ty), AREA, A, T,
                NN, E, lam, mu, clamped, sig);
      } else {
#pragma unroll
        for (int v = 0; v < NVOIGT; ++v)
          sig[v] = (real_t)0;
      }
#pragma unroll
      for (int v = 0; v < NVOIGT; ++v)
        ss[v * AREA + AT(x, ty)] = sig[v];
    }
    __syncthreads();

    // fd_kernel's update and the excitation; what nothing steps keeps its
    // value of two steps back, as the plain buffers do
#pragma unroll
    for (int k = 0; k < NODES; ++k) {
      const int x = x0 + k, a0 = lo0 - HALO + x;
      const int A[2] = {a0, a1};
      const int deep = warp_deep(A, NN);
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        real_t next = up[k][c];
        if ((moving >> (2 * k + c)) & 1) {
          const real_t force = BY_DEEP(deep, nodal_force, ss, AT(x, ty), AREA,
                                       T, c, A, NN, F, clamped);
#ifdef USE_DAMPING
          // rounded as nvcc compiles fd_kernel's, dt beta fused into 1 -+ beta
          next = FMA_RN(mi[k][c], force,
                        FMA_RN(-up[k][c], FMA_RN(-dt, beta[k][c], 1.f),
                               2.f * uc[k][c])) /
                 FMA_RN(dt, beta[k][c], 1.f);
#else
          next = -up[k][c] + 2.f * uc[k][c] + mi[k][c] * force;
#endif
          const int j = (tag[k][c] & 0xffffu) - 1;
          if (j >= 0)
            next +=
                MUL_RN(weights[j], signal[(size_t)(t0 + t) * num_sources + j]);
        }
        up[k][c] = uc[k][c];
        uc[k][c] = next;
        su[c * AREA + AT(x, ty)] = next;
        if (tag[k][c] >> 16)
          record[(size_t)(t0 + t) * num_sensors + (tag[k][c] >> 16) - 1] = next;
      }
    }
    __syncthreads();
  }

  // write back the core, each ghost with its mirror's tile
  if (!col || !core1)
    return;
#pragma unroll
  for (int k = 0; k < NODES; ++k) {
    const int a0 = lo0 - HALO + x0 + k;
    const int k0 = owner(a0, N0);
    if (a0 >= 0 && a0 <= N0 - 1 && k0 >= lo0 && k0 < lo0 + C0) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        v0[c * cs + a0 * s0 + a1] = up[k][c];
        v1[c * cs + a0 * s0 + a1] = uc[k][c];
      }
    }
  }
}
#endif

} // extern "C"
