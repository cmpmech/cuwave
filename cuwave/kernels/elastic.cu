// Compile-time configuration (set via -D flags from elastic.py):
//   USE_FLOAT
//   NDIM = 1 | 2 | 3
//   USE_DAMPING
//   STENCIL_RADIUS
//   STAG_COEFFS

#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

// ----------------------------- staggered difference helpers
#ifndef STENCIL_RADIUS
#define STENCIL_RADIUS 1 // default order 2
#endif

#if STENCIL_RADIUS == 1
#define SG_W(r, k) ((real_t)1)
#else
// cache
__constant__ real_t SG_C[STENCIL_RADIUS][STENCIL_RADIUS] = STAG_COEFFS;
#define SG_W(r, k) SG_C[(r) - 1][(k) - 1] // 1-indexed adjustment
#endif

#define NPAIRS (NDIM * (NDIM - 1) / 2)
#define NVOIGT (NDIM + NPAIRS)
#if NDIM == 3
#define PAIR_ROW(k, l) (NDIM + 3 - (k) - (l))
#else
#define PAIR_ROW(k, l) 2
#endif

// radius of the node strain along its own axis, zero on the wall itself
__device__ __forceinline__ int rad_node(const int a, const int N) {
  return min(STENCIL_RADIUS, min(a - 1, N - 2 - a));
}

// radius of a half-point derivative, whose taps may sit on the wall
__device__ __forceinline__ int rad_half(const int a, const int N) {
  return min(STENCIL_RADIUS, min(a, N - 2 - a));
}

__device__ __forceinline__ bool clamped_face(const int clamped, const int d,
                                             const int side) {
  return (clamped >> (2 * d + side)) & 1;
}

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

// the node's normal strains, one per axis: a wall grades to a zero row under
// traction (returned as a set bit) and to the antisymmetric fold when clamped
__device__ __forceinline__ int
normal_strains(const real_t *__restrict__ u, const int idx, const int cs,
               const int *A, const int *S, const int *NN, const real_t *F,
               const int clamped, real_t *eps) {
  int zeroed = 0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int r = rad_node(A[d], NN[d]);
    const real_t *__restrict__ ud = u + d * cs;
    if (r > 0) {
      real_t acc = (real_t)0;
#pragma unroll
      for (int k = 1; k <= STENCIL_RADIUS; ++k)
        if (k <= r)
          acc += SG_W(r, k) * (ud[idx + (k - 1) * S[d]] - ud[idx - k * S[d]]);
      eps[d] = F[d] * acc;
    } else if (A[d] == 1 && clamped_face(clamped, d, 0)) {
      eps[d] = F[d] * 2.f * ud[idx]; // the wall holds u = 0 half a node down
    } else if (A[d] == NN[d] - 2 && clamped_face(clamped, d, 1)) {
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
__device__ __forceinline__ real_t
shear_strain(const real_t *__restrict__ u, const int idx, const int cs,
             const int k, const int l, const int *A, const int *S,
             const int *NN, const real_t *F) {
  const int r1 = rad_half(A[l], NN[l]); // d u_k / d x_l
  const int r2 = rad_half(A[k], NN[k]); // d u_l / d x_k
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

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
stress_kernel(const real_t *__restrict__ u1, real_t *__restrict__ sigma,
              const real_t *__restrict__ gnode,
#if NDIM >= 2
              const real_t *__restrict__ gshear,
#endif
              const real_t lam, const real_t mu, const int clamped,
              const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

  real_t eps[NDIM];
  const int zeroed = normal_strains(u1, idx, cs, A, S, NN, F, clamped, eps);
  const real_t lam_eff = condensed_lame(lam, mu, zeroed);
  real_t tr = (real_t)0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    tr += eps[d];
  const real_t gn = gnode[idx];
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    sigma[d * cs + idx] = ((zeroed >> d) & 1)
                              ? (real_t)0
                              : gn * (2.f * mu * eps[d] + lam_eff * tr);

#if NDIM >= 2
#pragma unroll
  for (int k = 0; k < NDIM - 1; ++k)
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l) {
      if (A[k] > NN[k] - 3 || A[l] > NN[l] - 3)
        continue;
      const int v = PAIR_ROW(k, l);
      sigma[v * cs + idx] = gshear[(v - NDIM) * cs + idx] * mu *
                            shear_strain(u1, idx, cs, k, l, A, S, NN, F);
    }
#endif
}

// ------------------------------------------------------------------------------------
__global__ void
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ sigma,
          const real_t *__restrict__ minv,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
          const int clamped, const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;
  AXIS_FACTORS;

#pragma unroll
  for (int c = 0; c < NDIM; ++c) {
    if (A[c] > NN[c] - 3)
      continue; // no unknown on the staggered ghost of its own axis
    real_t force = (real_t)0;
    { // own normal stress along axis c, every tap keyed off the stress node
      const real_t *__restrict__ sc = sigma + c * cs;
      real_t acc = (real_t)0;
#pragma unroll
      for (int k = 1; k <= STENCIL_RADIUS; ++k) {
        const int ap = A[c] + k;
        const int rp = rad_node(ap, NN[c]);
        if (k <= rp)
          acc += SG_W(rp, k) * sc[idx + k * S[c]];
        else if (k == 1 && ap == NN[c] - 2 && clamped_face(clamped, c, 1))
          acc += 2.f * sc[idx + S[c]]; // the fold of the clamped wall strain
        const int am = A[c] - k + 1;
        const int rm = (am >= 1) ? rad_node(am, NN[c]) : 0;
        if (k <= rm)
          acc -= SG_W(rm, k) * sc[idx - (k - 1) * S[c]];
        else if (k == 1 && am == 1 && clamped_face(clamped, c, 0))
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
        const int rp = (ap <= NN[l] - 3) ? rad_half(ap, NN[l]) : 0;
        if (j <= rp)
          acc += SG_W(rp, j) * sv[idx + (j - 1) * S[l]];
        const int am = A[l] - j;
        const int rm = (am >= 1) ? rad_half(am, NN[l]) : 0;
        if (j <= rm)
          acc -= SG_W(rm, j) * sv[idx - j * S[l]];
      }
      force += F[l] * acc;
    }
#endif
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
__global__ void
excitation_kernel(real_t *__restrict__ u, const real_t *__restrict__ source,
                  const int offset, const int *__restrict__ lin_index,
                  const int num_sources, const real_t *__restrict__ weight) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sources) {
    atomicAdd(&u[lin_index[idx]], weight[idx] * source[offset + idx]);
  }
}

// ------------------------------------------------------------------------------------
__global__ void get_signal_kernel(const real_t *__restrict__ u,
                                  real_t *__restrict__ um, const int offset,
                                  const int *__restrict__ lin_index,
                                  const int num_sensors) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sensors) {
    um[offset + idx] = u[lin_index[idx]];
  }
}

// ------------------------------------------------------------------------------------
__global__ void set_signal_kernel(real_t *__restrict__ u,
                                  const real_t *__restrict__ um,
                                  const int offset,
                                  const int *__restrict__ lin_index,
                                  const int num_sensors) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sensors) {
    u[lin_index[idx]] = um[offset + idx];
  }
}

} // extern "C"
