// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 2 | 3
//   USE_DAMPING
//   USE_MAGNETIC
//   TILE0, TILE1, TILE2 (the fused step's tile)
//   STEPS, TILE0, TILE1, ROWS (2D, without USE_MAGNETIC)
//   GRADIENT (no STEPS): the step adds the gradient densities of u1 against v1

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

// the adjoint pair's accumulators and the forward field v1 they pair u1 with
#if defined(GRADIENT) && defined(USE_MAGNETIC)
#define CURL_GRADIENT , real_t *__restrict__ g_nu, const real_t *__restrict__ v1
#else
#define CURL_GRADIENT
#endif
#ifdef GRADIENT
#define MASS_GRADIENT                                                          \
  , real_t *__restrict__ g_mass, const real_t *__restrict__ v1, const real_t mf
#else
#define MASS_GRADIENT
#endif

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

// the force on component c: the curl rows of its pairs differenced along the
// other axis, strided `hs` a row and S an axis from idx
__device__ __forceinline__ real_t curl_force(const real_t *__restrict__ h,
                                             const int idx, const int hs,
                                             const int c, const int *A,
                                             const int *S, const int *NN,
                                             const real_t *F) {
  real_t force = (real_t)0;
#pragma unroll
  for (int m = 0; m < NDIM; ++m) { // the pairs this component belongs to
    if (m == c)
      continue;
    const int p = PAIR_ROW(min(c, m), max(c, m)) - NDIM;
    const real_t *__restrict__ hp = h + p * hs;
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
  return force;
}

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void curl_kernel(const real_t *__restrict__ u1,
                            real_t *__restrict__ h,
#ifdef USE_MAGNETIC
                            const real_t *__restrict__ nu_pair,
#endif
                            const real_t nu, const int cs,
                            AXIS_PARAMS CURL_GRADIENT) {
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
#if defined(GRADIENT) && defined(USE_MAGNETIC)
      // dJ/d(1/mu): the adjoint curl against the forward one
      g_nu[p * cs + idx] -= b * curl_component(v1, idx, cs, k, l, A, S, NN, F);
#endif
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
          const int cs, AXIS_PARAMS MASS_GRADIENT) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    const real_t force = curl_force(h, idx, cs, c, A, S, NN, F);
    const int n = c * cs + idx;
    const real_t mi = minv[n];
#ifdef USE_DAMPING
    const real_t beta = 0.5f * mi * damping[idx] * dt;
    const real_t next =
        (2.f * u1[n] - u0[n] * (1.f - beta) + mi * force) / (1.f + beta);
#else
    const real_t next = -u0[n] + 2.f * u1[n] + mi * force;
#endif
#ifdef GRADIENT
    // dJ/dpermittivity by parts in time: v against the second difference
    g_mass[n] -= mf * v1[n] * (next - 2.f * u1[n] + u0[n]);
#endif
    u2[n] = next;
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

#if defined(TILE0) && !defined(STEPS)
// ------------------------------------------------------------------------------------
// the curl of the block's tile and a ring a stencil radius wide around it
#define WIDE (TILE1 + 2 * STENCIL_RADIUS)
#if NDIM == 2
#define RING ((TILE0 + 2 * STENCIL_RADIUS) * WIDE)
#define THREADS (TILE0 * TILE1)
#else
#define DEEP_ (TILE2 + 2 * STENCIL_RADIUS)
#define RING ((TILE0 + 2 * STENCIL_RADIUS) * WIDE * DEEP_)
#define THREADS (TILE0 * TILE1 * TILE2)
#endif
__global__ void __launch_bounds__(THREADS)
    fused_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
                 real_t *__restrict__ u2,
#ifdef USE_MAGNETIC
                 const real_t *__restrict__ nu_pair,
#endif
                 const real_t nu, const real_t *__restrict__ minv,
#ifdef USE_DAMPING
                 const real_t *__restrict__ damping, const real_t dt,
#endif
                 const int cs, const real_t e0, const real_t e1,
#if NDIM == 3
                 const real_t e2,
#endif
                 AXIS_PARAMS MASS_GRADIENT
#if defined(GRADIENT) && defined(USE_MAGNETIC)
                 ,
                 real_t *__restrict__ g_nu
#endif
    ) {
  extern __shared__ real_t sh[]; // NPAIRS * RING
#if NDIM == 2
  const int b0 = blockIdx.y * TILE0 - STENCIL_RADIUS;
  const int b1 = blockIdx.x * TILE1 - STENCIL_RADIUS;
  const real_t E[2] = {e0, e1};
  for (int t = threadIdx.y * TILE1 + threadIdx.x; t < RING; t += THREADS) {
    const int g0 = b0 + t / WIDE, g1 = b1 + t % WIDE;
    const bool inner = g0 > 0 && g0 < N0 - 1 && g1 > 0 && g1 < N1 - 1;
    const int G[2] = {g0, g1}, GS[2] = {s0, 1}, GN[2] = {N0, N1};
    const int at = g0 * s0 + g1;
#else
  const int b0 = blockIdx.z * TILE0 - STENCIL_RADIUS;
  const int b1 = blockIdx.y * TILE1 - STENCIL_RADIUS;
  const int b2 = blockIdx.x * TILE2 - STENCIL_RADIUS;
  const real_t E[3] = {e0, e1, e2};
  for (int t = (threadIdx.z * TILE1 + threadIdx.y) * TILE2 + threadIdx.x;
       t < RING; t += THREADS) {
    const int g0 = b0 + t / (WIDE * DEEP_), g1 = b1 + t / DEEP_ % WIDE;
    const int g2 = b2 + t % DEEP_;
    const bool inner =
        g0 > 0 && g0 < N0 - 1 && g1 > 0 && g1 < N1 - 1 && g2 > 0 && g2 < N2 - 1;
    const int G[3] = {g0, g1, g2}, GS[3] = {s0, s1, 1}, GN[3] = {N0, N1, N2};
    const int at = g0 * s0 + g1 * s1 + g2;
#endif
    // curl_kernel's pair points; the scratch it writes is zero elsewhere
#pragma unroll
    for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
      for (int l = k + 1; l < NDIM; ++l) {
        const int p = PAIR_ROW(k, l) - NDIM;
        real_t h = (real_t)0;
        if (inner && G[k] <= GN[k] - 3 && G[l] <= GN[l] - 3) {
          const real_t b = curl_component(u1, at, cs, k, l, G, GS, GN, E);
#ifdef USE_MAGNETIC
          h = nu_pair[p * cs + at] * b;
#else
          h = nu * pair_weight(k, l, G, GN) * b;
#endif
        }
        sh[p * RING + t] = h;
      }
  }
  __syncthreads();

  // fd_kernel's update, the force read off the tile
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;
#if NDIM == 2
  const int local =
      (threadIdx.y + STENCIL_RADIUS) * WIDE + threadIdx.x + STENCIL_RADIUS;
  const int T[2] = {WIDE, 1};
#else
  const int local =
      ((threadIdx.z + STENCIL_RADIUS) * WIDE + threadIdx.y + STENCIL_RADIUS) *
          DEEP_ +
      threadIdx.x + STENCIL_RADIUS;
  const int T[3] = {WIDE * DEEP_, DEEP_, 1};
#endif
#if defined(GRADIENT) && defined(USE_MAGNETIC)
  // dJ/d(1/mu) on the node's pair points, with the curl's factors E
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      if (A[k] > NN[k] - 3 || A[l] > NN[l] - 3)
        continue;
      g_nu[(PAIR_ROW(k, l) - NDIM) * cs + idx] -=
          curl_component(u1, idx, cs, k, l, A, S, NN, E) *
          curl_component(v1, idx, cs, k, l, A, S, NN, E);
    }
#endif
#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    const real_t force = curl_force(sh, local, RING, c, A, T, NN, F);
    const int n = c * cs + idx;
    const real_t mi = minv[n];
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

#if NDIM == 2 && defined(STEPS) && !defined(USE_MAGNETIC)
// ------------------------------------------------------------------------------------
// the core a tile writes back, its halo STEPS curl-force reaches wide
#define HALO (STEPS * (2 * STENCIL_RADIUS - 1))
#define C0 (TILE0 - 2 * HALO)
#define C1 (TILE1 - 2 * HALO)
#define NODES (TILE0 / ROWS) // a thread's consecutive nodes along axis 0
#define WIDE (TILE1 + 2 * STENCIL_RADIUS) // a tile row with its guard columns
#define AREA ((TILE0 + 2 * STENCIL_RADIUS) * WIDE)
#define AT(x, y) (((x) + STENCIL_RADIUS) * WIDE + (y) + STENCIL_RADIUS)
static_assert(TILE0 % ROWS == 0, "the thread rows split the tile evenly");
static_assert(2 * NODES <= 32, "a component bit per node fits the mask");

// advance a tile STEPS steps between one load and one store: the curl, the
// update, the excitation and the record, bitwise the plain launches
__global__ void __launch_bounds__(TILE1 *ROWS)
    blocked_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
                   real_t *__restrict__ v0, real_t *__restrict__ v1,
                   const real_t *__restrict__ minv, const real_t nu,
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
                   const int cs, const real_t e0, const real_t e1,
                   const real_t f0, const int N0, const real_t f1, const int N1,
                   const int s0) {
  // both components of E and the curl, each in a guard ring
  extern __shared__ real_t shared[];
  real_t *su = shared, *sh = shared + 2 * AREA;
  const int ty = threadIdx.x, x0 = threadIdx.y * NODES;
  const int thread = threadIdx.y * TILE1 + ty;
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  const int lo0 = 1 + blockIdx.y * C0, lo1 = 1 + blockIdx.x * C1;
  const int a1 = lo1 - HALO + ty, k1 = owner(a1, N1);
  const bool col = a1 >= 0 && a1 <= N1 - 1;
  const bool core1 = k1 >= lo1 && k1 < lo1 + C1;
  const real_t E[2] = {e0, e1}, F[2] = {f0, f1};
  const int T[2] = {WIDE, 1}, NN[2] = {N0, N1};
  for (int e = thread; e < 3 * AREA; e += TILE1 * ROWS)
    shared[e] = 0;
  __syncthreads();

  // the tile's sources and sensors as tags in the E planes, source + 1 and
  // sensor + 1 a half, the component in the entry's local node
  unsigned *tags = (unsigned *)su;
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

  // the thread's nodes, loaded once: E twice, 1 / m and beta per component
  real_t up[NODES][2], uc[NODES][2], mi[NODES][2];
#ifdef USE_DAMPING
  real_t beta[NODES][2];
#endif
  unsigned paired = 0, moving = 0; // bit k: a curl point; 2 k + c: c steps
#pragma unroll
  for (int k = 0; k < NODES; ++k) {
    const int x = x0 + k, a0 = lo0 - HALO + x, idx = a0 * s0 + a1;
    const bool inside = col && a0 >= 0 && a0 <= N0 - 1;
    const bool interior = a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1;
    const int A[2] = {a0, a1};
    paired |= (unsigned)(interior && a0 <= N0 - 3 && a1 <= N1 - 3) << k;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      up[k][c] = inside ? u0[c * cs + idx] : 0;
      uc[k][c] = inside ? u1[c * cs + idx] : 0;
      su[c * AREA + AT(x, ty)] = uc[k][c];
      const bool steps = interior && A[c] <= NN[c] - 3;
      moving |= (unsigned)steps << (2 * k + c);
      mi[k][c] = steps ? minv[c * cs + idx] : (real_t)0;
#ifdef USE_DAMPING
      beta[k][c] = steps ? 0.5f * mi[k][c] * damping[idx] : 0; // over dt
#endif
    }
  }
  __syncthreads();

  for (int t = 0; t < STEPS; ++t) {
    // curl_kernel's pair points, the tile edge forming garbage that never
    // reaches the core
#pragma unroll
    for (int k = 0; k < NODES; ++k) {
      const int x = x0 + k, a0 = lo0 - HALO + x;
      const int A[2] = {a0, a1};
      real_t b = (real_t)0;
      if ((paired >> k) & 1)
        b = nu * pair_weight(0, 1, A, NN) *
            curl_component(su, AT(x, ty), AREA, 0, 1, A, T, NN, E);
      sh[AT(x, ty)] = b;
    }
    __syncthreads();

    // fd_kernel's update and the excitation; what nothing steps keeps its
    // value of two steps back, as the plain buffers do
#pragma unroll
    for (int k = 0; k < NODES; ++k) {
      const int x = x0 + k, a0 = lo0 - HALO + x;
      const int A[2] = {a0, a1};
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        real_t next = up[k][c];
        if ((moving >> (2 * k + c)) & 1) {
          const real_t force = curl_force(sh, AT(x, ty), AREA, c, A, T, NN, F);
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
