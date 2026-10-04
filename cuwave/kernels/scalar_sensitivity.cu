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
#define AXIS_RADII const int r0 = CLOSURE(a0, N0), r1 = CLOSURE(a1, N1)
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
#define AXIS_RADII                                                             \
  const int r0 = CLOSURE(a0, N0), r1 = CLOSURE(a1, N1), r2 = CLOSURE(a2, N2)
#define AXIS_OFFSETS const int o0 = s0, o1 = s1, o2 = 1
#define AXIS_ARRAYS(F)                                                         \
  const int a[3] = {a0, a1, a2}, n[3] = {N0, N1, N2}, s[3] = {s0, s1, 1};      \
  const real_t f[3] = {F##0, F##1, F##2}
#endif

// ------------------------------------ split launch
// a row the node gathers from is graded, halved or a ghost, or a ghost reads it
#define WALL_REACH (STENCIL_RADIUS > 1 ? 2 * STENCIL_RADIUS : 3)
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
__device__ __forceinline__ real_t stiffness_gradient_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ l1,
    const real_t *__restrict__ stiff, const int idx, const int s,
    const real_t uc, const real_t lc, const real_t sc, const real_t factor) {
  const real_t sp = stiff[idx + s];                     // plus of sc
  const real_t sm = stiff[idx - s];                     // minus of sc
  const real_t dgp = sp * sp / ((sc + sp) * (sc + sp)); // d(harmonic mean)/dsc
  const real_t dgm = sm * sm / ((sc + sm) * (sc + sm)); // d(harmonic mean)/dsc
  real_t Dp = OP_W(STENCIL_RADIUS, 1) * (u1[idx + s] - uc); // inner grad
  real_t Dm = OP_W(STENCIL_RADIUS, 1) * (uc - u1[idx - s]); // inner grad
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k) {
    Dp += OP_W(STENCIL_RADIUS, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]);
    Dm += OP_W(STENCIL_RADIUS, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]);
  }
  return factor * (dgp * Dp * (l1[idx + s] - lc) +
                   dgm * Dm * (lc - l1[idx - s])); // both cells of the node
}

// ---------------------------- transposed operator helpers
// the transposed flux divergence off the walls: compact rise, wide divergence
__device__ __forceinline__ real_t transposed_divergence_axis(
    const real_t *__restrict__ l1, const real_t *__restrict__ stiff,
    const int idx, const int s, const real_t lc, const real_t sc) {
  real_t div = 0, sh = sc, sl = sc, lh = lc, ll = lc; // inner ends of the cells
#pragma unroll
  for (int k = 1; k <= STENCIL_RADIUS; ++k) {
    const real_t sh1 = stiff[idx + k * s], sl1 = stiff[idx - k * s];
    const real_t lh1 = l1[idx + k * s], ll1 = l1[idx - k * s];
    div += OP_W(STENCIL_RADIUS, k) * (sh * sh1 / (sh + sh1) * (lh1 - lh) -
                                      sl * sl1 / (sl + sl1) * (ll - ll1));
    sh = sh1, sl = sl1, lh = lh1, ll = ll1;
  }
  return div;
}

// ------------------------------------ wall helpers
// a row's distance to the nearer ghost, capped past the deepest graded row
__device__ __forceinline__ int row_distance(const int i, const int N) {
  return max(0, min(min(i, N - 1 - i), STENCIL_RADIUS + 1));
}

// tap k of a row at distance j, its graded radius and cell weight folded in
__device__ __forceinline__ real_t graded_tap(const int j, const int k) {
  if (j < k)
    return 0;
  return (j == 1 ? (real_t)0.5 : (real_t)1) * OP_W(min(STENCIL_RADIUS, j), k);
}

__device__ __forceinline__ real_t cell_mean(const real_t sl, const real_t sh) {
  return sl * sh / (sl + sh); // harmonic mean
}

// row a + i of a line, clamped into it
__device__ __forceinline__ real_t line(const real_t *__restrict__ field,
                                       const int idx, const int s, const int a,
                                       const int N, const int i) {
  return field[idx + (min(max(a + i, 0), N - 1) - a) * s];
}

// tap k's flux of the cell between rows a + i and a + i + 1, loaded per call
__device__ __forceinline__ real_t graded_flux(const real_t *__restrict__ l1,
                                              const real_t *__restrict__ stiff,
                                              const int idx, const int s,
                                              const int a, const int N,
                                              const int i, const int k) {
  return cell_mean(line(stiff, idx, s, a, N, i),
                   line(stiff, idx, s, a, N, i + 1)) *
         (graded_tap(row_distance(a + i + 1, N), k) *
              line(l1, idx, s, a, N, i + 1) -
          graded_tap(row_distance(a + i, N), k) * line(l1, idx, s, a, N, i));
}

// the general row, loading per tap so the deep path keeps its registers
__device__ __forceinline__ real_t graded_divergence(
    const real_t *__restrict__ l1, const real_t *__restrict__ stiff,
    const int idx, const int s, const int a, const int N, const int faces) {
  real_t div = 0;
#pragma unroll
  for (int k = 1; k <= STENCIL_RADIUS; ++k) // the cells k out on either side
    div += graded_flux(l1, stiff, idx, s, a, N, k - 1, k) -
           graded_flux(l1, stiff, idx, s, a, N, -k, k);
  if (a == 2 || a == N - 3) {
    // the ghost column folded onto its mirror, odd on a Dirichlet face
    const int o = a == 2 ? -s : s, odd = a == 2 ? faces & 1 : faces & 2;
    real_t fold = 0;
#pragma unroll
    for (int k = 1; k <= STENCIL_RADIUS; ++k) // rows k and k - 1 off the ghost
      fold += cell_mean(stiff[idx + (2 - k) * o], stiff[idx + (3 - k) * o]) *
              graded_tap(row_distance(a == 2 ? k : N - 1 - k, N), k) *
              l1[idx + (2 - k) * o];
    div += odd ? -fold : fold;
  }
  return row_distance(a, N) == 1 ? 2 * div : div; // over the wall's half cell
}

__device__ __forceinline__ real_t
graded_gradient(const real_t *__restrict__ u1, const real_t *__restrict__ l1,
                const real_t *__restrict__ stiff, const int idx, const int s,
                const int a, const int N) {
  const int jm = row_distance(a - 1, N), jc = row_distance(a, N);
  const int jp = row_distance(a + 1, N);
  const real_t qm = line(l1, idx, s, a, N, -1), qc = l1[idx];
  const real_t qp = line(l1, idx, s, a, N, 1);
  real_t sum_p = 0, sum_m = 0; // flux of u times rise of lambda, per cell
#pragma unroll
  for (int k = 1; k <= STENCIL_RADIUS; ++k) {
    sum_p += (line(u1, idx, s, a, N, k) - line(u1, idx, s, a, N, 1 - k)) *
             (graded_tap(jp, k) * qp - graded_tap(jc, k) * qc);
    sum_m += (line(u1, idx, s, a, N, k - 1) - line(u1, idx, s, a, N, -k)) *
             (graded_tap(jc, k) * qc - graded_tap(jm, k) * qm);
  }
  const real_t sc = stiff[idx], sp = stiff[idx + s], sm = stiff[idx - s];
  real_t g = sp * sp / ((sc + sp) * (sc + sp)) * sum_p + // d(harmonic mean)/dsc
             sm * sm / ((sc + sm) * (sc + sm)) * sum_m;
  if (a == 2 || a == N - 3) {
    // the ghost's stiffness is its mirror's, and its one cell has radius 1
    const int o = a == 2 ? -s : s;
    const real_t sg = stiff[idx + 2 * o], sw = stiff[idx + o];
    g += sw * sw / ((sg + sw) * (sg + sw)) * (real_t)0.5 * l1[idx + o] *
         (u1[idx + o] - u1[idx + 2 * o]);
  }
  return jc == 1 ? 2 * g : g; // over the wall's half cell
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
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      g += NEAR_WALL(a[d], n[d])
               ? f[d] * graded_gradient(u1, l1, stiff, idx, s[d], a[d], n[d])
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
  if (!deep) {
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      div_l += f[d] *
               (NEAR_WALL(a[d], n[d])
                    ? graded_divergence(l1, stiff, idx, s[d], a[d], n[d],
                                        dirichlet >> 2 * d)
                    : transposed_divergence_axis(l1, stiff, idx, s[d], lc, sc));
  } else {
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      div_l += f[d] * transposed_divergence_axis(l1, stiff, idx, s[d], lc, sc);
  }

#ifdef USE_DAMPING
  l2[idx] = (2.f * lc - lo * (1.f - beta) + mi * div_l) / (1.f + beta);
#else
  l2[idx] = -lo + 2.f * lc + mi * div_l;
#endif
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
  if (!deep) {
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      if (NEAR_WALL(a[d], n[d])) {
        div_l += f[d] * graded_divergence(l1, stiff, idx, s[d], a[d], n[d],
                                          dirichlet >> 2 * d);
        g += f[d] * graded_gradient(u1, l1, stiff, idx, s[d], a[d], n[d]);
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
  const real_t ln = (2.f * lc - lo * (1.f - beta) + mi * div_l) / (1.f + beta);
#else
  const real_t ln = -lo + 2.f * lc + mi * div_l;
#endif
  l0[idx] = ln;
  // dJ/dmass by parts in time: u against the second difference of lambda
  g_mass[idx] = gm - mf * uc * (ln - 2.f * lc + lo);
  g_stiff[idx] = gs - mf * g;
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
