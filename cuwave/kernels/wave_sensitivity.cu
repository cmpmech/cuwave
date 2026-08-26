// Discrete-adjoint gradient kernels for the pressure-wave discretisation in
// wave.cu (read that file first: it defines the recursion, the material fields
// and the compile-time flags, all of which are shared verbatim).
//
// The forward step is  u^t = 2u^{t-1} - u^{t-2} + m (L u^{t-1}) + m b^t  with
// m = minv dt^2 and L the flux-form div(stiff grad .). Making the Lagrangian
// J(u) + sum_t psi^t . C~^t stationary (C~ = the residual divided by m, which
// symmetrises it) leaves the same recursion run backwards, so the adjoint field
// lambda needs no kernel of its own -- sensitivity.py steps it with fd_kernel.
// What is left is dJ/dtheta = sum_t psi^t d(C~^t)/dtheta, which gradient_kernel
// accumulates into one array per material field, for the driver to combine by
// chain rule:
//
//   dJ/dmass  = -1/dt^2 sum_t lambda^t (u^t - 2u^{t-1} + u^{t-2})     mass = 1/minv
//   dJ/dstiff = -sum_t sum_k F_k [ dgp Dp (l[+s] - l) + dgm Dm (l - l[-s]) ]
//
// Both are exact derivatives of the discretisation, not of the PDE. Summed by
// parts in time the first is the familiar Frechet kernel int (du/dt)(dl/dt), but
// the form here needs no time differencing at all. The second differentiates
// wave.cu's harmonic face mean in place: dgp / dgm are d(sc sn / (sc + sn))/dsc
// on the two faces of the node, and Dp / Dm are the very same face differences
// flux_divergence_axis already forms. That reuse is what makes it exact at every
// space_order rather than only at order 2 -- the face-pairing argument behind the
// formula relies on Dm at node j+s being identically Dp at node j, which holds
// for the telescoped weights at any radius.
//
// Both accumulate into a per-node scalar over the padded grid, so a driver may
// call the kernel for a subset of the time steps or with aliased fields.

#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

#ifndef STENCIL_RADIUS  // sensitivity.py prepends the table, as wave.py does for
#define STENCIL_RADIUS 1  // wave.cu; this keeps the file standalone-compilable
#endif

#if STENCIL_RADIUS == 1
#define OP_W(r, k) ((real_t)1)
#define CLOSURE(a, N) 1
#else
__constant__ real_t OP_C[STENCIL_RADIUS][STENCIL_RADIUS] = OP_COEFFS;
#define OP_W(r, k) OP_C[(r) - 1][(k) - 1]
#define CLOSURE(a, N) min(STENCIL_RADIUS, min(a, (N) - 1 - (a)))
#endif

// Every kernel here opens with the same interior guard and flat index, and needs
// either the graded stencil radii or the neighbour offsets alongside it. Macros
// rather than helpers because they declare the axis indices and return early. The
// fastest axis maps to x, exactly as grid_block does on the host.
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

// One axis of d/d(stiff[idx]) of lambda . (L u), summed over the two faces the
// node owns. Mirrors flux_divergence_axis in wave.cu term for term: same loads,
// same Dp / Dm loop, with the face coefficient replaced by its derivative and
// the outer lambda[idx] replaced by the face differences of lambda.
__device__ __forceinline__ real_t stiffness_gradient_axis(
    const real_t *__restrict__ u1, const real_t *__restrict__ l1,
    const real_t *__restrict__ stiff, const int idx, const int s,
    const real_t uc, const real_t lc, const real_t sc, const real_t factor,
    const int r) {
  const real_t sp = stiff[idx + s];
  const real_t sm = stiff[idx - s];
  const real_t dgp = sp * sp / ((sc + sp) * (sc + sp));
  const real_t dgm = sm * sm / ((sc + sm) * (sc + sm));
  real_t Dp = OP_W(r, 1) * (u1[idx + s] - uc);
  real_t Dm = OP_W(r, 1) * (uc - u1[idx - s]);
#pragma unroll
  for (int k = 2; k <= STENCIL_RADIUS; ++k)
    if (k <= r) {
      Dp += OP_W(r, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]);
      Dm += OP_W(r, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]);
    }
  return factor * (dgp * Dp * (l1[idx + s] - lc) + dgm * Dm * (lc - l1[idx - s]));
}

extern "C" {

// The two gradients stay separate *outputs* -- separate accumulators, separate
// chain-rule factors at the call site -- but they are accumulated by one kernel
// rather than two. A launch costs more CPU time than either body costs on the
// device (see the note above define_step_method in wave.py), and fusing also loads
// u1, lambda and the interior guard once instead of twice. dJ/dmass is node-local;
// dJ/dstiff needs the neighbours of u1, lambda and stiff.
//
// The per-axis factor is wave.cu's f_k with the dt^2 divided out, i.e.
// F_k = 2 c^2 / dx_k^2 (scalar) or 2 / dx_k^2 (acoustic): L itself, not the dt^2 L
// that the step applies.

__global__ void gradient_kernel(real_t *__restrict__ g_mass,
                                real_t *__restrict__ g_stiff,
                                const real_t *__restrict__ u0,
                                const real_t *__restrict__ u1,
                                const real_t *__restrict__ u2,
                                const real_t *__restrict__ l1,
                                const real_t *__restrict__ stiff,
                                const real_t inv_dt2, const real_t F0,
                                const int N0
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

  // dJ/dstiff: wave.cu's harmonic face mean differentiated in place
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

} // extern "C"

// ======================================================================================
// Frechet-form integrands for the superposition variant (superposition.py).
//
// The trick evaluates a bilinear form on the diagonal: with w = u + k lambda,
//   B(w, w) - B(u, u) = 2 k B(u, lambda) + k^2 B(lambda, lambda)
// which only holds if B is *symmetric*, so it cannot reuse the exact-transpose
// kernels above -- those pair a wide Dp(u) stencil against a single difference of
// lambda. These two use the symmetric central-difference form instead,
//   B_mass(a, b) = sum_n (da/dt)(db/dt),  B_stiff(a, b) = sum_n grad a . grad b,
// which is the Frechet kernel of the FWI literature: consistent with the operator
// to O(dx^2, dt^2) rather than its exact transpose. Only the diagonal is ever
// needed, so it takes one field triplet, not two -- half the loads of a bilinear
// version, and the whole point of the trick.
//
// Note both terms are invariant under time reversal: the backward pass walks the triplet
// the other way, which flips the sign of du/dt and leaves its square alone. That is
// what lets the reconstructed forward field and the adjoint field share one array.
//
// ft and f_k carry the +/-1 that subtracts the forward diagonal and adds the
// superposed one, with the 1/(2 dt)^2 and 1/(2 dx_k)^2 folded in on the host.

extern "C" {

__global__ void frechet_kernel(real_t *__restrict__ acc_mass,
                               real_t *__restrict__ acc_stiff,
                               const real_t *__restrict__ u0,
                               const real_t *__restrict__ u1,
                               const real_t *__restrict__ u2,
                               const real_t ft, const real_t f0, const int N0
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

} // extern "C"
