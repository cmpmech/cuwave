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
#define AXIS_OFFSETS const int o0 = 1
#define AXIS_ARRAYS(F)                                                         \
  const int a[1] = {a0}, n[1] = {N0}, s[1] = {1};                              \
  const real_t f[1] = {F##0}
#elif NDIM == 2
#define INTERIOR_OR_RETURN                                                     \
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;                        \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1))                       \
    return;                                                                    \
  const int idx = a0 * s0 + a1
#define AXIS_OFFSETS const int o0 = s0, o1 = 1
#define AXIS_ARRAYS(F)                                                         \
  const int a[2] = {a0, a1}, n[2] = {N0, N1}, s[2] = {s0, 1};                  \
  const real_t f[2] = {F##0, F##1}
#elif NDIM == 3
#define INTERIOR_OR_RETURN                                                     \
  const int a2 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  const int a1 = blockIdx.y * blockDim.y + threadIdx.y;                        \
  const int a0 = blockIdx.z * blockDim.z + threadIdx.z;                        \
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&            \
        a2 < N2 - 1))                                                          \
    return;                                                                    \
  const int idx = a0 * s0 + a1 * s1 + a2
#define AXIS_OFFSETS const int o0 = s0, o1 = s1, o2 = 1
#define AXIS_ARRAYS(F)                                                         \
  const int a[3] = {a0, a1, a2}, n[3] = {N0, N1, N2}, s[3] = {s0, s1, 1};      \
  const real_t f[3] = {F##0, F##1, F##2}
#endif

// ------------------------------------ split launch
// a row whose taps reach a wall node or past it, read through the mirror
#define WALL_REACH (STENCIL_RADIUS + 2)
#define NEAR_WALL(a, N) ((a) < WALL_REACH || (a) > (N) - 1 - WALL_REACH)
#define WALL_COLUMNS (2 * (WALL_REACH - 1)) // near-wall nodes of a fast row

// whether a block's span [lo, lo + n) of an axis holds no near-wall row
#define DEEP_SPAN(lo, n, N) (!NEAR_WALL(lo, N) && !NEAR_WALL((lo) + (n) - 1, N))

// the leading blocks of each block row, packing every row's near-wall columns
__device__ __forceinline__ int wall_blocks() {
  return (WALL_COLUMNS + blockDim.x - 1) / blockDim.x;
}

// a wall block's thread, counted across all of them so they pack densely
__device__ __forceinline__ int wall_thread() {
  const int w =
      (blockIdx.z * gridDim.y + blockIdx.y) * wall_blocks() + blockIdx.x;
  const int t = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x;
  return w * blockDim.x * blockDim.y * blockDim.z + t + threadIdx.x;
}

// near-wall column c of a fast row, or -1 where the low side already holds it
__device__ __forceinline__ int wall_column(const int c, const int X) {
  const int high = c >= WALL_REACH - 1;
  const int ax = high ? X - 2 * WALL_REACH + 1 + c : 1 + c;
  return (high ? ax < WALL_REACH : ax > X - 2) ? -1 : ax;
}

// a deep block's fast-axis node, or -1 past the deep columns
__device__ __forceinline__ int deep_column(const int X) {
  const int ax =
      WALL_REACH + (blockIdx.x - wall_blocks()) * blockDim.x + threadIdx.x;
  return ax < X - WALL_REACH ? ax : -1;
}

#if NDIM == 1
#define SPLIT_OR_RETURN                                                        \
  const bool deep = blockIdx.x >= wall_blocks();                               \
  const int t = wall_thread();                                                 \
  const int a0 = deep               ? deep_column(N0)                          \
                 : t < WALL_COLUMNS ? wall_column(t, N0)                       \
                                    : -1;                                      \
  if (a0 < 0)                                                                  \
    return;                                                                    \
  const int idx = a0
#elif NDIM == 2
#define SPLIT_OR_RETURN                                                        \
  const bool deep_x = blockIdx.x >= wall_blocks();                             \
  const int t = wall_thread();                                                 \
  const int a1 = deep_x ? deep_column(N1) : wall_column(t % WALL_COLUMNS, N1); \
  const int a0 =                                                               \
      deep_x ? blockIdx.y * blockDim.y + threadIdx.y : t / WALL_COLUMNS;       \
  if (a1 < 0 || !(a0 > 0 && a0 < N0 - 1))                                      \
    return;                                                                    \
  const bool deep =                                                            \
      deep_x && DEEP_SPAN(blockIdx.y * blockDim.y, blockDim.y, N0);            \
  const int idx = a0 * s0 + a1
#elif NDIM == 3
#define SPLIT_OR_RETURN                                                        \
  const bool deep_x = blockIdx.x >= wall_blocks();                             \
  const int t = wall_thread(), r = t / WALL_COLUMNS;                           \
  const int a2 = deep_x ? deep_column(N2) : wall_column(t % WALL_COLUMNS, N2); \
  const int a1 = deep_x ? blockIdx.y * blockDim.y + threadIdx.y : r % N1;      \
  const int a0 = deep_x ? blockIdx.z * blockDim.z + threadIdx.z : r / N1;      \
  if (a2 < 0 || !(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1))             \
    return;                                                                    \
  const bool deep = deep_x &&                                                  \
                    DEEP_SPAN(blockIdx.y * blockDim.y, blockDim.y, N1) &&      \
                    DEEP_SPAN(blockIdx.z * blockDim.z, blockDim.z, N0);        \
  const int idx = a0 * s0 + a1 * s1 + a2
#endif

// ---------------------------------- gradient helpers
// d(stiffness)/dsc of the two cells, `u(j)`, `l(j)`, `k(j)` the taps j on
template <class U, class L, class K>
__device__ __forceinline__ real_t
stiffness_gradient_taps(const U &u, const L &l, const K &k, const real_t uc,
                        const real_t lc, const real_t sc, const real_t factor) {
  const real_t sp = k(1);                               // plus of sc
  const real_t sm = k(-1);                              // minus of sc
  const real_t dgp = sp * sp / ((sc + sp) * (sc + sp)); // d(harmonic mean)/dsc
  const real_t dgm = sm * sm / ((sc + sm) * (sc + sm)); // d(harmonic mean)/dsc
  real_t Dp = OP_W(STENCIL_RADIUS, 1) * (u(1) - uc);    // inner grad
  real_t Dm = OP_W(STENCIL_RADIUS, 1) * (uc - u(-1));   // inner grad
#pragma unroll
  for (int j = 2; j <= STENCIL_RADIUS; ++j) {
    Dp += OP_W(STENCIL_RADIUS, j) * (u(j) - u(1 - j));
    Dm += OP_W(STENCIL_RADIUS, j) * (u(j - 1) - u(-j));
  }
  return factor * (dgp * Dp * (l(1) - lc) +
                   dgm * Dm * (lc - l(-1))); // both cells of the node
}

// ---------------------------- transposed operator helpers
// the transposed flux divergence: compact rise, wide divergence
template <class L, class K>
__device__ __forceinline__ real_t transposed_divergence_taps(const L &l,
                                                             const K &k,
                                                             const real_t lc,
                                                             const real_t sc) {
  real_t div = 0, sh = sc, sl = sc, lh = lc, ll = lc; // inner ends of the cells
#pragma unroll
  for (int j = 1; j <= STENCIL_RADIUS; ++j) {
    const real_t sh1 = k(j), sl1 = k(-j);
    const real_t lh1 = l(j), ll1 = l(-j);
    div += OP_W(STENCIL_RADIUS, j) * (sh * sh1 / (sh + sh1) * (lh1 - lh) -
                                      sl * sl1 / (sl + sl1) * (ll - ll1));
    sh = sh1, sl = sl1, lh = lh1, ll = ll1;
  }
  return div;
}

// both along axis s off the walls, the taps read in place
__device__ __forceinline__ real_t stiffness_gradient_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ l1,
    const real_t *__restrict__ stiff, const int idx, const int s,
    const real_t uc, const real_t lc, const real_t sc, const real_t factor) {
  const auto u = [&](const int j) { return u1[idx + j * s]; };
  const auto l = [&](const int j) { return l1[idx + j * s]; };
  const auto k = [&](const int j) { return stiff[idx + j * s]; };
  return stiffness_gradient_taps(u, l, k, uc, lc, sc, factor);
}

__device__ __forceinline__ real_t transposed_divergence_axis(
    const real_t *__restrict__ l1, const real_t *__restrict__ stiff,
    const int idx, const int s, const real_t lc, const real_t sc) {
  const auto l = [&](const int j) { return l1[idx + j * s]; };
  const auto k = [&](const int j) { return stiff[idx + j * s]; };
  return transposed_divergence_taps(l, k, lc, sc);
}

// ------------------------------------ wall helpers
// node_imaged, never a ghost the adjoint leaves unfilled, zero on a held node
__device__ __forceinline__ real_t reflected(const real_t *__restrict__ u,
                                            const int idx, const int s,
                                            const int a, const int N,
                                            const int odd, const int j) {
  int sign;
  const int b = node_image(a + j, N, odd, sign);
  return signed_value(u[idx + (b - a) * s], held_wall(b, N, odd) ? 0 : sign);
}

// the same near a wall, node a of N, every tap through its mirror: lambda and
// u odd on a face of `odd`, zero on its wall node, the stiffness even
__device__ __forceinline__ real_t mirrored_gradient(
    const real_t *__restrict__ u1, const real_t *__restrict__ l1,
    const real_t *__restrict__ stiff, const int idx, const int s, const int a,
    const int N, const int odd, const real_t uc, const real_t lc,
    const real_t sc, const real_t factor) {
  const auto u = [&](const int j) {
    return reflected(u1, idx, s, a, N, odd, j);
  };
  const auto l = [&](const int j) {
    return reflected(l1, idx, s, a, N, odd, j);
  };
  const auto k = [&](const int j) {
    return reflected(stiff, idx, s, a, N, 0, j);
  };
  return stiffness_gradient_taps(u, l, k, uc, lc, sc, factor);
}

__device__ __forceinline__ real_t
mirrored_divergence(const real_t *__restrict__ l1,
                    const real_t *__restrict__ stiff, const int idx,
                    const int s, const int a, const int N, const int odd,
                    const real_t lc, const real_t sc) {
  const auto l = [&](const int j) {
    return reflected(l1, idx, s, a, N, odd, j);
  };
  const auto k = [&](const int j) {
    return reflected(stiff, idx, s, a, N, 0, j);
  };
  return transposed_divergence_taps(l, k, lc, sc);
}

// ----------------------------- pml transposed helpers
// d(Dp)/d(u at row q off the node)
__device__ __forceinline__ real_t plus_tap(const int q) {
  constexpr int r = STENCIL_RADIUS;
  if (q >= 1)
    return q <= r ? OP_W(r, q) : (real_t)0;
  return 1 - q <= r ? -OP_W(r, 1 - q) : (real_t)0;
}

// d(Dm)/d(u at row q off the node)
__device__ __forceinline__ real_t minus_tap(const int q) {
  constexpr int r = STENCIL_RADIUS;
  if (q >= 0)
    return q + 1 <= r ? OP_W(r, q + 1) : (real_t)0;
  return -q <= r ? -OP_W(r, -q) : (real_t)0;
}

// the memory of layer row ai (slab thread i) read back off the row q from it
__device__ __forceinline__ real_t pml_tap(const real_t *__restrict__ aux,
                                          const real_t *__restrict__ stiff,
                                          const real_t *__restrict__ profile,
                                          const real_t f, const int N,
                                          const int o, const int size,
                                          const int i, const int iidx,
                                          const int ai, const int q) {
  const real_t *an = profile, *ah = profile + 2 * N;
  const real_t sc = stiff[iidx], sp = stiff[iidx + o], sm = stiff[iidx - o];
  const real_t gp = sc * sp / (sc + sp), gm = sc * sm / (sc + sm); // harmonic
  const real_t ez = an[ai] * f * aux[2 * size + i];
  const real_t cp = ah[ai] * aux[i] + gp * (1 + ah[ai]) * ez;
  const real_t cm = ah[ai - 1] * aux[size + i] - gm * (1 + ah[ai - 1]) * ez;
  return cp * plus_tap(q) + cm * minus_tap(q);
}

// byte-identical to scalar.cu: each file is its own compilation unit
// spatial finite difference stencil for Laplacian, `u(j)` the tap j nodes on
template <class Taps>
__device__ __forceinline__ real_t flux_divergence_taps(
    const Taps &u, const real_t uc, const real_t gp, const real_t gm,
    const real_t factor, const int r) {
  // rounded as nvcc contracts the plain sums, so every caller agrees bitwise
  real_t Dp = u(1) - uc, Dm = uc - u(-1); // inner grad
  if (STENCIL_RADIUS > 1) { // a weight of one is exact, and an intrinsic stays
    Dp = MUL_RN(OP_W(r, 1), Dp);
    Dm = MUL_RN(OP_W(r, 1), Dm);
  }
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Dp = FMA_RN(OP_W(r, k), u(k) - u(1 - k), Dp);
      Dm = FMA_RN(OP_W(r, k), u(k - 1) - u(-k), Dm);
    }
  return factor * FMA_RN(Dp, gp, -MUL_RN(Dm, gm)); // outer grad
}

// spatial finite difference stencil for Laplacian, node a of N on its axis
__device__ __forceinline__ real_t flux_divergence_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ stiff,
    const int idx, const int s, const real_t uc, const real_t sc, const int a,
    const int N, const int odd) {
  const real_t sp = stiff[idx + s]; // plus of sc
  const real_t sm = stiff[idx - s]; // minus of sc
  // every tap through the mirror, a no-op inside, so no warp diverges on it
  real_t taps[2 * STENCIL_RADIUS + 1];
#pragma unroll
  for (int j = -STENCIL_RADIUS; j <= STENCIL_RADIUS; ++j)
    taps[j + STENCIL_RADIUS] = mirrored(u1, idx, s, a, N, odd, j);
  const auto u = [&](const int j) { return taps[j + STENCIL_RADIUS]; };
  return flux_divergence_taps(u, uc, sc * sp / (sc + sp), sc * sm / (sc + sm),
                              1, STENCIL_RADIUS);
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
  SPLIT_OR_RETURN;
  AXIS_ARRAYS(F);

  const real_t uc = u1[idx];
  const real_t lc = l1[idx];

  // dJ/dmass: no neighbour and no material load
  g_mass[idx] -= inv_dt2 * lc * (u2[idx] - 2.f * uc + u0[idx]);

  // dJ/dstiff: scalar.cu's harmonic cell mean differentiated in place
  const real_t sc = stiff[idx];
  real_t g = 0;
  if (!deep) {
    // lambda reads zero on a Dirichlet wall node, its own included
    bool held = false;
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      held |= held_wall(a[d], n[d], DIRICHLET_FACES >> 2 * d);
    const real_t lw = held ? (real_t)0 : lc;
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      g += NEAR_WALL(a[d], n[d])
               ? mirrored_gradient(u1, l1, stiff, idx, s[d], a[d], n[d],
                                   DIRICHLET_FACES >> 2 * d, uc, lw, sc, f[d])
               : stiffness_gradient_axis(u1, l1, stiff, idx, s[d], uc, lc, sc,
                                         f[d]);
  } else {
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      g += stiffness_gradient_axis(u1, l1, stiff, idx, s[d], uc, lc, sc, f[d]);
  }
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
__global__ void
adjoint_kernel(const real_t *__restrict__ l0, const real_t *__restrict__ l1,
               real_t *__restrict__ l2, const real_t *__restrict__ stiff,
               const real_t *__restrict__ minv, const int derive_inertia,
#ifdef USE_DAMPING
               const real_t *__restrict__ damping, const real_t dt,
#endif
               const int dirichlet, const real_t f0, const int N0
#if NDIM >= 2
               ,
               const real_t f1, const int N1, const int s0
#endif
#if NDIM >= 3
               ,
               const real_t f2, const int N2, const int s1
#endif
) {
  SPLIT_OR_RETURN;
  AXIS_ARRAYS(f);

  // the node's own loads, which both paths share
  const real_t lc = l1[idx];
  const real_t sc = stiff[idx];
  const real_t lo = l0[idx];
  const real_t mi = derive_inertia ? 1.f / sc : minv[idx];
#ifdef USE_DAMPING
  const real_t beta = 0.5f * mi * damping[idx] * dt;
#endif
  real_t div_l = 0;
  bool held = false;
  if (!deep) {
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      held |= held_wall(a[d], n[d], dirichlet >> 2 * d);
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      div_l += f[d] *
               (NEAR_WALL(a[d], n[d])
                    ? mirrored_divergence(l1, stiff, idx, s[d], a[d], n[d],
                                          dirichlet >> 2 * d, lc, sc)
                    : transposed_divergence_axis(l1, stiff, idx, s[d], lc, sc));
  } else {
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      div_l += f[d] * transposed_divergence_axis(l1, stiff, idx, s[d], lc, sc);
  }

#ifdef USE_DAMPING
  real_t ln = (2.f * lc - lo * (1.f - beta) + mi * div_l) / (1.f + beta);
#else
  real_t ln = -lo + 2.f * lc + mi * div_l;
#endif
  l2[idx] = held ? (real_t)0 : ln; // a Dirichlet wall node holds zero
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
    const int dirichlet, const real_t mf, const real_t f0, const int N0
#if NDIM >= 2
    ,
    const real_t f1, const int N1, const int s0
#endif
#if NDIM >= 3
    ,
    const real_t f2, const int N2, const int s1
#endif
) {
  SPLIT_OR_RETURN;
  AXIS_ARRAYS(f);

  // the node's own loads, which both paths share
  const real_t uc = u1[idx];
  const real_t lc = l1[idx];
  const real_t sc = stiff[idx];
  const real_t lo = l0[idx];
  const real_t gm = g_mass[idx], gs = g_stiff[idx];
  const real_t mi = derive_inertia ? 1.f / sc : minv[idx];
#ifdef USE_DAMPING
  const real_t beta = 0.5f * mi * damping[idx] * dt;
#endif
  real_t div_l = 0, g = 0;
  bool held = false;
  if (!deep) {
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      held |= held_wall(a[d], n[d], dirichlet >> 2 * d);
    const real_t lw = held ? (real_t)0 : lc; // lambda reads zero there
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      if (NEAR_WALL(a[d], n[d])) {
        const int odd = dirichlet >> 2 * d;
        div_l += f[d] * mirrored_divergence(l1, stiff, idx, s[d], a[d], n[d],
                                            odd, lc, sc);
        g += mirrored_gradient(u1, l1, stiff, idx, s[d], a[d], n[d], odd, uc,
                               lw, sc, f[d]);
      } else {
        div_l +=
            f[d] * transposed_divergence_axis(l1, stiff, idx, s[d], lc, sc);
        g +=
            stiffness_gradient_axis(u1, l1, stiff, idx, s[d], uc, lc, sc, f[d]);
      }
  } else {
#pragma unroll
    for (int d = 0; d < NDIM; ++d) {
      div_l += f[d] * transposed_divergence_axis(l1, stiff, idx, s[d], lc, sc);
      g += stiffness_gradient_axis(u1, l1, stiff, idx, s[d], uc, lc, sc, f[d]);
    }
  }

#ifdef USE_DAMPING
  real_t ln = (2.f * lc - lo * (1.f - beta) + mi * div_l) / (1.f + beta);
#else
  real_t ln = -lo + 2.f * lc + mi * div_l;
#endif
  if (held)
    ln = 0; // a Dirichlet wall node holds zero
  l0[idx] = ln;
  // dJ/dmass by parts in time: u against the second difference of lambda
  g_mass[idx] = gm - mf * uc * (ln - 2.f * lc + lo);
  g_stiff[idx] = gs - mf * g;
}

// ------------------------------------------------------------------------------------
__global__ void pml_aux_kernel(const real_t *__restrict__ l1,
                               const real_t *__restrict__ stiff,
                               real_t *__restrict__ aux,
                               const real_t *__restrict__ profile, const int d,
                               const int lo, const int hi, const real_t f,
                               PML_PARAMS) {
  PML_GEOM;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  int idx, a, off;
  real_t w;
  if (!slab_node(t, d, lo, hi, n, s, idx, a, off, w))
    return;
  const int N = n[d], o = s[d], size = (lo + hi) * slab_plane(d, n, s);
  const real_t *an = profile, *bn = profile + N, *bh = profile + 3 * N;

  const real_t sc = stiff[idx], sp = stiff[idx + o], sm = stiff[idx - o];
  const real_t gp = sc * sp / (sc + sp), gm = sc * sm / (sc + sm); // harmonic
  // lambda / m, the stepped field being that over W, zero where it is held
  const real_t q = held_node(idx, n, s) ? (real_t)0 : w * l1[idx];
  const real_t ez = aux[2 * size + t];
  aux[t] = bh[a] * (aux[t] + an[a] * f * gp * ez) + f * gp * q;
  aux[size + t] =
      bh[a - 1] * (aux[size + t] - an[a] * f * gm * ez) - f * gm * q;
  aux[2 * size + t] = bn[a] * ez + q;
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_kernel(
    real_t *__restrict__ l2, real_t *__restrict__ g_mass,
    const real_t *__restrict__ u1, const real_t mf, const int with_mass,
    const real_t *__restrict__ stiff, const real_t *__restrict__ minv,
    const int derive_inertia, const real_t *__restrict__ aux,
    const real_t *__restrict__ profile, const int d, const int lo, const int hi,
    const real_t f, const int dirichlet, PML_PARAMS) {
  PML_GEOM;
  // every node the layer's taps reach, up to STENCIL_RADIUS past its front
  const int elo = lo ? lo + STENCIL_RADIUS : 0;
  const int ehi = hi ? hi + STENCIL_RADIUS : 0;
  int idx, a, off;
  real_t w;
  if (!slab_node(blockIdx.x * blockDim.x + threadIdx.x, d, elo, ehi, n, s, idx,
                 a, off, w))
    return;
  const int N = n[d], o = s[d], plane = slab_plane(d, n, s);
  const int size = (lo + hi) * plane;

  real_t sum = 0;
#pragma unroll
  for (int m = -STENCIL_RADIUS; m <= STENCIL_RADIUS; ++m) {
    const int i = slab_index(a + m, off, d, lo, hi, n, plane);
    if (i >= 0)
      sum += pml_tap(aux, stiff, profile, f, N, o, size, i, idx + m * o, a + m,
                     -m);
  }
#pragma unroll
  for (int high = 0; high < 2; ++high) {
    // the reads past the wall that land on a, folded back, odd on a Dirichlet
    // face
    const int ghost = high ? 2 * (N - 2) - a : 2 - a;
    if (high ? ghost < N - 1 : ghost > 0)
      continue;
    real_t fold = 0;
#pragma unroll
    for (int k = 1; k <= STENCIL_RADIUS; ++k) {
      const int ai = high ? N - 1 - k : k;
      const int i = slab_index(ai, off, d, lo, hi, n, plane);
      if (i >= 0 && ghost - ai >= -STENCIL_RADIUS &&
          ghost - ai <= STENCIL_RADIUS)
        fold += pml_tap(aux, stiff, profile, f, N, o, size, i,
                        idx + (ai - a) * o, ai, ghost - ai);
    }
    sum += (dirichlet >> (2 * d + high)) & 1 ? -fold : fold;
  }

  const real_t mi = derive_inertia ? 1.f / stiff[idx] : minv[idx];
  const real_t delta = mi * sum / w;
  l2[idx] += delta;
  if (with_mass) // the fused step read lambda^n before this share of it
    g_mass[idx] -= mf * u1[idx] * delta;
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
  AXIS_OFFSETS;
  constexpr int odd = DIRICHLET_FACES;

  const real_t uc = u1[idx];
  const real_t sc = stiff[idx];
  // the axes summed as fd_kernel sums them, so the field steps bitwise alike
#if NDIM == 1
  const real_t laplacian =
      MUL_RN(f0, flux_divergence_axis(u1, stiff, idx, 1, uc, sc, a0, N0, odd));
  const bool held = held_wall(a0, N0, odd);
#elif NDIM == 2
  const real_t d0 =
      flux_divergence_axis(u1, stiff, idx, s0, uc, sc, a0, N0, odd);
  const real_t d1 =
      flux_divergence_axis(u1, stiff, idx, 1, uc, sc, a1, N1, odd >> 2);
#if STENCIL_RADIUS == 1
  const real_t laplacian = FMA_RN(f0, d0, MUL_RN(f1, d1));
#else
  const real_t laplacian = FMA_RN(f1, d1, MUL_RN(f0, d0));
#endif
  const bool held = held_wall(a0, N0, odd) || held_wall(a1, N1, odd >> 2);
#elif NDIM == 3
  const real_t d0 =
      flux_divergence_axis(u1, stiff, idx, s0, uc, sc, a0, N0, odd);
  const real_t d1 =
      flux_divergence_axis(u1, stiff, idx, s1, uc, sc, a1, N1, odd >> 2);
  const real_t d2 =
      flux_divergence_axis(u1, stiff, idx, 1, uc, sc, a2, N2, odd >> 4);
  const real_t laplacian = FMA_RN(f2, d2, FMA_RN(f0, d0, MUL_RN(f1, d1)));
  const bool held = held_wall(a0, N0, odd) || held_wall(a1, N1, odd >> 2) ||
                    held_wall(a2, N2, odd >> 4);
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
  u2[idx] = held ? (real_t)0 : -u0[idx] + 2.f * uc + mi * laplacian;
}

} // extern "C"
