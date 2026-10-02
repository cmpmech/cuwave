// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3
//   USE_DAMPING

// ------------------------------- interior guard macros
#if NDIM == 1
#define INTERIOR_OR_RETURN                                                     \
  const int a0 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  if (!(a0 > 0 && a0 < N0 - 1))                                                \
    return;                                                                    \
  const int idx = a0
#define AXIS_RADII const int r0 = CLOSURE(a0, N0)
#define AXIS_OFFSETS const int o0 = 1
#elif NDIM == 2
#define INTERIOR_OR_RETURN                                                     \
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;                        \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1))                       \
    return;                                                                    \
  const int idx = a0 * s0 + a1
#define AXIS_RADII const int r0 = CLOSURE(a0, N0), r1 = CLOSURE(a1, N1)
#define AXIS_OFFSETS const int o0 = s0, o1 = 1
#elif NDIM == 3
#define INTERIOR_OR_RETURN                                                     \
  const int a2 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  const int a1 = blockIdx.y * blockDim.y + threadIdx.y;                        \
  const int a0 = blockIdx.z * blockDim.z + threadIdx.z;                        \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&            \
        a2 < N2 - 1))                                                          \
    return;                                                                    \
  const int idx = a0 * s0 + a1 * s1 + a2
#define AXIS_RADII                                                             \
  const int r0 = CLOSURE(a0, N0), r1 = CLOSURE(a1, N1), r2 = CLOSURE(a2, N2)
#define AXIS_OFFSETS const int o0 = s0, o1 = s1, o2 = 1
#endif

// ---------------------------------- gradient helpers
__device__ __forceinline__ real_t stiffness_gradient_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ l1,
    const real_t *__restrict__ stiff, const int idx, const int s,
    const real_t uc, const real_t lc, const real_t sc, const real_t factor,
    const int r) {
  const real_t sp = stiff[idx + s];                     // plus of sc
  const real_t sm = stiff[idx - s];                     // minus of sc
  const real_t dgp = sp * sp / ((sc + sp) * (sc + sp)); // d(harmonic mean)/dsc
  const real_t dgm = sm * sm / ((sc + sm) * (sc + sm)); // d(harmonic mean)/dsc
  real_t Dp = OP_W(r, 1) * (u1[idx + s] - uc);          // inner grad
  real_t Dm = OP_W(r, 1) * (uc - u1[idx - s]);          // inner grad
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Dp += OP_W(r, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]);
      Dm += OP_W(r, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]);
    }
  return factor * (dgp * Dp * (l1[idx + s] - lc) +
                   dgm * Dm * (lc - l1[idx - s])); // both cells of the node
}

// the adjoint step's flux divergence and the stiffness gradient of one axis,
// sharing the cell stiffnesses
__device__ __forceinline__ void adjoint_gradient_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ l1,
    const real_t *__restrict__ stiff, const int idx, const int s,
    const real_t uc, const real_t lc, const real_t sc, const real_t factor,
    const int r, real_t &div_l, real_t &g) {
  const real_t sp = stiff[idx + s];                     // plus of sc
  const real_t sm = stiff[idx - s];                     // minus of sc
  const real_t gp = sc * sp / (sc + sp);                // harmonic mean
  const real_t gm = sc * sm / (sc + sm);                // harmonic mean
  const real_t dgp = sp * sp / ((sc + sp) * (sc + sp)); // d(harmonic mean)/dsc
  const real_t dgm = sm * sm / ((sc + sm) * (sc + sm)); // d(harmonic mean)/dsc
  const real_t lp = l1[idx + s];
  const real_t lm = l1[idx - s];
  real_t Lp = OP_W(r, 1) * (lp - lc);          // inner grad of lambda
  real_t Lm = OP_W(r, 1) * (lc - lm);          // inner grad of lambda
  real_t Dp = OP_W(r, 1) * (u1[idx + s] - uc); // inner grad of u
  real_t Dm = OP_W(r, 1) * (uc - u1[idx - s]); // inner grad of u
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Lp += OP_W(r, k) * (l1[idx + k * s] - l1[idx - (k - 1) * s]);
      Lm += OP_W(r, k) * (l1[idx + (k - 1) * s] - l1[idx - k * s]);
      Dp += OP_W(r, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]);
      Dm += OP_W(r, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]);
    }
  div_l += factor * (Lp * gp - Lm * gm); // the step's flux divergence
  g += factor * (dgp * Dp * (lp - lc) + dgm * Dm * (lc - lm));
}

// byte-identical to scalar.cu: each file is its own compilation unit
__device__ __forceinline__ real_t flux_divergence_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ stiff,
    const int idx, const int s, const real_t uc, const real_t sc,
    const real_t factor, const int r) {
  const real_t sp = stiff[idx + s];            // plus of sc
  const real_t sm = stiff[idx - s];            // minus of sc
  const real_t gp = sc * sp / (sc + sp);       // harmonic mean
  const real_t gm = sc * sm / (sc + sm);       // harmonic mean
  real_t Dp = OP_W(r, 1) * (u1[idx + s] - uc); // initialization: inner grad
  real_t Dm = OP_W(r, 1) * (uc - u1[idx - s]); // initialization: inner grad
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Dp +=
          OP_W(r, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]); // inner grad
      Dm +=
          OP_W(r, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]); // inner grad
    }
  return factor * (Dp * gp - Dm * gm); // outer grad (incl. inner grad)
}

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
gradient_kernel(real_t *__restrict__ g_mass, real_t *__restrict__ g_stiff,
                const real_t *__restrict__ u0, const real_t *__restrict__ u1,
                const real_t *__restrict__ u2, const real_t *__restrict__ l1,
                const real_t *__restrict__ stiff, const real_t inv_dt2,
                const real_t F0, const int N0
#if NDIM >= 2
                ,
                const real_t F1, const int N1, const int s0
#endif
#if NDIM >= 3
                ,
                const real_t F2, const int N2, const int s1
#endif
) {
  INTERIOR_OR_RETURN;
  AXIS_RADII;

  const real_t uc = u1[idx];
  const real_t lc = l1[idx];

  // dJ/dmass: no neighbour and no material load
  g_mass[idx] -= inv_dt2 * lc * (u2[idx] - 2.f * uc + u0[idx]);

  // dJ/dstiff: scalar.cu's harmonic cell mean differentiated in place
  const real_t sc = stiff[idx];
#if NDIM == 1
  const real_t g =
      stiffness_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, F0, r0);
#elif NDIM == 2
  const real_t g =
      stiffness_gradient_axis(u1, l1, stiff, idx, s0, uc, lc, sc, F0, r0) +
      stiffness_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, F1, r1);
#elif NDIM == 3
  const real_t g =
      stiffness_gradient_axis(u1, l1, stiff, idx, s0, uc, lc, sc, F0, r0) +
      stiffness_gradient_axis(u1, l1, stiff, idx, s1, uc, lc, sc, F1, r1) +
      stiffness_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, F2, r2);
#endif
  g_stiff[idx] -= g;
}

// ------------------------------------------------------------------------------------
__global__ void frechet_kernel(real_t *__restrict__ acc_mass,
                               real_t *__restrict__ acc_stiff,
                               const real_t *__restrict__ u0,
                               const real_t *__restrict__ u1,
                               const real_t *__restrict__ u2, const real_t ft,
                               const real_t f0, const int N0
#if NDIM >= 2
                               ,
                               const real_t f1, const int N1, const int s0
#endif
#if NDIM >= 3
                               ,
                               const real_t f2, const int N2, const int s1
#endif
) {
  INTERIOR_OR_RETURN;
  AXIS_OFFSETS;

  const real_t dudt = u2[idx] - u0[idx];
  acc_mass[idx] += ft * dudt * dudt;

  const real_t g0 = u1[idx + o0] - u1[idx - o0];
  real_t sum = f0 * g0 * g0;
#if NDIM >= 2
  const real_t g1 = u1[idx + o1] - u1[idx - o1];
  sum += f1 * g1 * g1;
#endif
#if NDIM >= 3
  const real_t g2 = u1[idx + o2] - u1[idx - o2];
  sum += f2 * g2 * g2;
#endif
  acc_stiff[idx] += sum;
}

// ------------------------------------------------------------------------------------
__global__ void adjoint_gradient_kernel(
    real_t *__restrict__ l0, const real_t *__restrict__ l1,
    real_t *__restrict__ g_mass, real_t *__restrict__ g_stiff,
    const real_t *__restrict__ u1, const real_t *__restrict__ stiff,
    const real_t *__restrict__ minv, const int derive_inertia,
#ifdef USE_DAMPING
    const real_t *__restrict__ damping, const real_t dt,
#endif
    const real_t mf, const real_t f0, const int N0
#if NDIM >= 2
    ,
    const real_t f1, const int N1, const int s0
#endif
#if NDIM >= 3
    ,
    const real_t f2, const int N2, const int s1
#endif
) {
  INTERIOR_OR_RETURN;
  AXIS_RADII;

  const real_t uc = u1[idx];
  const real_t lc = l1[idx];
  const real_t sc = stiff[idx];
  real_t div_l = 0, g = 0;
#if NDIM == 1
  adjoint_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, f0, r0, div_l, g);
#elif NDIM == 2
  adjoint_gradient_axis(u1, l1, stiff, idx, s0, uc, lc, sc, f0, r0, div_l, g);
  adjoint_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, f1, r1, div_l, g);
#elif NDIM == 3
  adjoint_gradient_axis(u1, l1, stiff, idx, s0, uc, lc, sc, f0, r0, div_l, g);
  adjoint_gradient_axis(u1, l1, stiff, idx, s1, uc, lc, sc, f1, r1, div_l, g);
  adjoint_gradient_axis(u1, l1, stiff, idx, 1, uc, lc, sc, f2, r2, div_l, g);
#endif

  const real_t mi = derive_inertia ? 1.f / sc : minv[idx];
  const real_t lo = l0[idx];
#ifdef USE_DAMPING
  const real_t beta = 0.5f * mi * damping[idx] * dt;
  const real_t ln = (2.f * lc - lo * (1.f - beta) + mi * div_l) / (1.f + beta);
#else
  const real_t ln = -lo + 2.f * lc + mi * div_l;
#endif
  l0[idx] = ln;
  // dJ/dmass by parts in time: u against the second difference of lambda
  g_mass[idx] -= mf * uc * (ln - 2.f * lc + lo);
  g_stiff[idx] -= mf * g;
}

// ------------------------------------------------------------------------------------
__global__ void superposed_kernel(
    const real_t *__restrict__ u0, const real_t *__restrict__ u1,
    real_t *__restrict__ u2, real_t *__restrict__ acc_mass,
    real_t *__restrict__ acc_stiff, const real_t *__restrict__ stiff,
    const real_t *__restrict__ minv, const int derive_inertia, const real_t ft,
    const real_t fs, const real_t f0, const int N0
#if NDIM >= 2
    ,
    const real_t f1, const int N1, const int s0
#endif
#if NDIM >= 3
    ,
    const real_t f2, const int N2, const int s1
#endif
) {
  INTERIOR_OR_RETURN;
  AXIS_RADII;
  AXIS_OFFSETS;

  const real_t uc = u1[idx];
  const real_t sc = stiff[idx];
#if NDIM == 1
  const real_t laplacian =
      flux_divergence_axis(u1, stiff, idx, 1, uc, sc, f0, r0);
#elif NDIM == 2
  const real_t laplacian =
      flux_divergence_axis(u1, stiff, idx, s0, uc, sc, f0, r0) +
      flux_divergence_axis(u1, stiff, idx, 1, uc, sc, f1, r1);
#elif NDIM == 3
  const real_t laplacian =
      flux_divergence_axis(u1, stiff, idx, s0, uc, sc, f0, r0) +
      flux_divergence_axis(u1, stiff, idx, s1, uc, sc, f1, r1) +
      flux_divergence_axis(u1, stiff, idx, 1, uc, sc, f2, r2);
#endif

  // the mass density of the triplet before, whose oldest slot u2 still holds
  const real_t dudt = uc - u2[idx];
  acc_mass[idx] += ft * dudt * dudt;

  // the stiffness density of this triplet, its middle slot being u1
  const real_t g0 = u1[idx + o0] - u1[idx - o0];
  real_t sum = f0 * g0 * g0;
#if NDIM >= 2
  const real_t g1 = u1[idx + o1] - u1[idx - o1];
  sum += f1 * g1 * g1;
#endif
#if NDIM >= 3
  const real_t g2 = u1[idx + o2] - u1[idx - o2];
  sum += f2 * g2 * g2;
#endif
  acc_stiff[idx] += fs * sum;

  const real_t mi = derive_inertia ? 1.f / sc : minv[idx];
  u2[idx] = -u0[idx] + 2.f * uc + mi * laplacian;
}

} // extern "C"
