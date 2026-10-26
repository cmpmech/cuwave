// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3
//   USE_DAMPING
//   USE_PML
//   TILE0, TILE1 (2D, no pml): the fused step's tile

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

// G times the strain terms t of axis e, added onto the node's stresses sig
__device__ __forceinline__ void
add_stress(real_t *sig, const real_t *__restrict__ gamma, const int idx,
           const int e, const real_t *t, const int *A, const int *S,
           const int *NN, const real_t lam, const real_t mu,
           const int clamped) {
  const int zeroed = zeroed_axes(A, NN, clamped);
  if (!((zeroed >> e) & 1)) {
    const real_t lam_eff = condensed_lame(lam, mu, zeroed);
    const real_t g = node_gamma(gamma, idx, A, NN) * t[e];
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
          shear_gamma(gamma, idx, min(k, e), max(k, e), A, S, NN) * mu * t[k];
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
             const int *S, const int *NN, const real_t *F, const int clamped) {
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
    const real_t term = divergence_term(k, c, e, A, NN, F[e], clamped);
    if (t >= 0) // the divergence memory, read by the stress pass before
      *z = profile[NN[e] + at] * (*z + u1[c * cs + idx]);
    return term;
  }
  const Stress sig = {sigma, idx, cs, S[e], e};
  *z = profile[NN[e] + at] * *z +
       profile[at] * divergence_term(sig, c, e, A, NN, F[e], clamped);
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

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
stress_kernel(const real_t *__restrict__ u1, real_t *__restrict__ sigma,
              const real_t *__restrict__ gamma, const real_t lam,
              const real_t mu, const int clamped, const int cs,
              AXIS_PARAMS PML_FOLD) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

  real_t sig[NVOIGT];
  if (deep_node(A, NN))
    stress_point<true>(u1, gamma, idx, cs, A, S, NN, F, lam, mu, clamped, sig);
  else
    stress_point(u1, gamma, idx, cs, A, S, NN, F, lam, mu, clamped, sig);
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
__global__ void
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ sigma,
          const real_t *__restrict__ gamma, const real_t rho,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
          const int clamped, const int cs, AXIS_PARAMS PML_FOLD) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;
#ifdef USE_PML
  FOLD_GEOM;
#endif

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
                                  HI[e], phase, A, S, NN, F, clamped);
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

#if NDIM == 2 && defined(TILE0) && !defined(USE_PML)
// ------------------------------------------------------------------------------------
// the stresses of the block's tile and a ring a stencil radius wide around it
#define RING (TILE0 + 2 * STENCIL_RADIUS)
#define WIDE (TILE1 + 2 * STENCIL_RADIUS)
__global__ void __launch_bounds__(TILE0 *TILE1)
    fused_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
                 real_t *__restrict__ u2, const real_t *__restrict__ gamma,
                 const real_t lam, const real_t mu, const real_t rho,
#ifdef USE_DAMPING
                 const real_t *__restrict__ damping, const real_t dt,
#endif
                 const int clamped, const int cs, const real_t e0,
                 const real_t e1, AXIS_PARAMS) {
  __shared__ real_t ss[NVOIGT * RING * WIDE];
  const int b0 = blockIdx.y * TILE0 - STENCIL_RADIUS;
  const int b1 = blockIdx.x * TILE1 - STENCIL_RADIUS;
  const real_t E[2] = {e0, e1};
  for (int t = threadIdx.y * TILE1 + threadIdx.x; t < RING * WIDE;
       t += TILE0 * TILE1) {
    const int g0 = b0 + t / WIDE, g1 = b1 + t % WIDE;
    real_t sig[NVOIGT];
    // stress_kernel's nodes; the scratch it writes is zero elsewhere
    if (g0 > 0 && g0 < N0 - 1 && g1 > 0 && g1 < N1 - 1) {
      const int G[2] = {g0, g1}, GS[2] = {s0, 1}, GN[2] = {N0, N1};
      const int at = g0 * s0 + g1;
      if (deep_node(G, GN))
        stress_point<true>(u1, gamma, at, cs, G, GS, GN, E, lam, mu, clamped,
                           sig);
      else
        stress_point(u1, gamma, at, cs, G, GS, GN, E, lam, mu, clamped, sig);
    } else {
#pragma unroll
      for (int v = 0; v < NVOIGT; ++v)
        sig[v] = (real_t)0;
    }
#pragma unroll
    for (int v = 0; v < NVOIGT; ++v)
      ss[v * RING * WIDE + t] = sig[v];
  }
  __syncthreads();

  // fd_kernel's update, the divergence read off the tile
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;
  const int local = (threadIdx.y + STENCIL_RADIUS) * WIDE + threadIdx.x +
                    STENCIL_RADIUS;
  const int T[2] = {WIDE, 1};
  const bool deep = deep_node(A, NN);
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    const real_t force =
        deep ? nodal_force<true>(ss, local, RING * WIDE, T, c, A, NN, F,
                                 clamped)
             : nodal_force(ss, local, RING * WIDE, T, c, A, NN, F, clamped);
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
#endif

} // extern "C"
