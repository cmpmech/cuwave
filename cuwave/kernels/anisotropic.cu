// Prepended by wave.compile_kernels: stencils.preamble, then common.cuh.
// Compile-time configuration this file responds to:
//   NDIM = 1 | 2 | 3
//   USE_DAMPING

#ifndef RADIUS
#define RADIUS 1 // default order 2
#endif

// the node block a full-radius cell reaches, and the cells a node borders
#define BLK (2 * RADIUS)
#if NDIM == 1
#define CELLS BLK
#elif NDIM == 2
#define CELLS (BLK * BLK)
#else
#define CELLS (BLK * BLK * BLK)
#endif
#define NLOC (CELLS * NDIM)

// position along axis d of the block entry m, axis 0 running fastest
__device__ __forceinline__ int block_axis(int m, const int d) {
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
    if (k < d)
      m /= BLK;
  return m % BLK;
}

// ----------------------------- cell assembly helpers
#if NDIM == 1
#define AXIS_PARAMS const real_t f0, const int N0
#define INTERIOR_OR_RETURN                                                     \
  const int a0 = blockIdx.x * blockDim.x + threadIdx.x;                        \
  if (!(a0 > 0 && a0 < N0 - 1))                                                \
    return;                                                                    \
  const int idx = a0
#define AXIS_GEOM const int A[1] = {a0}, S[1] = {1}, NN[1] = {N0}
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
#endif

// A cell sits inside the domain when both its own corners do. Its radius
// grades down towards a wall so the stencil never passes a ghost node, and
// the gathering node must fall inside that graded support to take from it.
__device__ __forceinline__ bool cell_inside(const int c, const int *A,
                                            const int *NN, const int *S,
                                            int &base, int &radius,
                                            int &self) {
  base = 0;
  self = 0;
  radius = RADIUS;
  bool inside = true;
  int power = 1;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int e = block_axis(c, d) - RADIUS; // low corner, from the node
    const int low = A[d] + e;
    inside &= (low >= 1 && low <= NN[d] - 3);
    radius = min(radius, min(RADIUS, min(low, NN[d] - 2 - low)));
    base += e * S[d];
    self += (RADIUS - 1 - e) * power;
    power *= BLK;
  }
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int e = block_axis(c, d) - RADIUS;
    inside &= (e >= -radius && e <= radius - 1);
  }
  return inside;
}

// offset of block entry l of cell c from the gathering node, and whether
// that entry falls inside the cell's graded support at all
__device__ __forceinline__ bool block_offset(const int c, const int l,
                                             const int radius, const int *S,
                                             int &off) {
  off = 0;
  bool carries = true;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    const int m = block_axis(l, d);
    carries &= (m >= RADIUS - radius && m <= RADIUS + radius - 1);
    off += (block_axis(c, d) - RADIUS + m - RADIUS + 1) * S[d];
  }
  return carries;
}

// ------------------------------------------ pml helpers
// the uniform argument list of the pml kernels, so one launch list serves all
#define PML_ARGS                                                               \
  const real_t *__restrict__ u1, real_t *__restrict__ u2,                      \
      real_t *__restrict__ aux, const real_t *__restrict__ minv,               \
      const real_t *__restrict__ cell, const real_t *__restrict__ cv,          \
      const real_t f, const int cs, const real_t *__restrict__ profile,        \
      const int *__restrict__ layers, const int e, AXIS_PARAMS

// one lined axis: memory rows (cell strains, then node forces) and profile
struct Layer {
  real_t *aux;
  const real_t *profile;
  int lo, hi, plane, size;
};

__device__ __forceinline__ Layer layer_of(const int d, real_t *aux,
                                          const real_t *profile,
                                          const int *layers, const int *n,
                                          const int *s) {
  Layer l;
  l.lo = layers[4 * d];
  l.hi = layers[4 * d + 1];
  l.aux = aux + layers[4 * d + 2];
  l.profile = profile + layers[4 * d + 3];
  l.plane = slab_plane(d, n, s);
  l.size = (l.lo + l.hi) * l.plane;
  return l;
}

// the slab thread of the node at X, or of the cell with low corner X, or -1
__device__ __forceinline__ int slab_at(const int *X, const bool cell,
                                       const int d, const Layer &l,
                                       const int *n, const int *s) {
  if (!(l.lo + l.hi))
    return -1;
  int off = 0;
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
    if (k != d)
      off = off * padded_extent(k, n, s) + X[k];
  // a high side cell belongs to the node on its wall side
  const int a = (cell && X[d] > l.lo) ? X[d] + 1 : X[d];
  return slab_index(a, off, d, l.lo, l.hi, n, l.plane);
}

// centre-point derivative weight along d at corner m of a cell
__device__ __forceinline__ real_t corner_weight(const int m, const int d,
                                                const real_t *F) {
  return (((m >> d) & 1) ? F[d] : -F[d]) / (real_t)(1 << (NDIM - 1));
}

__device__ __forceinline__ int flat(const int *X, const int *S) {
  int n = 0;
#pragma unroll
  for (int d = 0; d < NDIM; ++d)
    n += X[d] * S[d];
  return n;
}

// the cell gradient G[k][j] = d u_j / d x_k at the centre, low corner X
__device__ __forceinline__ void cell_gradient(const real_t *__restrict__ u,
                                              const int *X, const int cs,
                                              const int *S, const real_t *F,
                                              real_t G[NDIM][NDIM]) {
  const int base = flat(X, S);
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
#pragma unroll
    for (int j = 0; j < NDIM; ++j)
      G[k][j] = (real_t)0;
#pragma unroll
  for (int m = 0; m < (1 << NDIM); ++m) {
    int n = base;
#pragma unroll
    for (int d = 0; d < NDIM; ++d)
      n += ((m >> d) & 1) * S[d];
#pragma unroll
    for (int j = 0; j < NDIM; ++j) {
      const real_t uj = u[j * cs + n];
#pragma unroll
      for (int k = 0; k < NDIM; ++k)
        G[k][j] += corner_weight(m, k, F) * uj;
    }
  }
}

// the stress g V C sym(G) in Voigt rows, the shear strains engineering
__device__ __forceinline__ void cell_stress(const real_t G[NDIM][NDIM],
                                            const real_t *__restrict__ cv,
                                            const real_t g,
                                            real_t sig[NVOIGT]) {
  real_t eps[NVOIGT];
#pragma unroll
  for (int k = 0; k < NDIM; ++k) {
    eps[k] = G[k][k];
#pragma unroll
    for (int l = k + 1; l < NDIM; ++l)
      eps[PAIR_ROW(k, l)] = G[k][l] + G[l][k];
  }
#pragma unroll
  for (int r = 0; r < NVOIGT; ++r) {
    real_t acc = (real_t)0;
#pragma unroll
    for (int q = 0; q < NVOIGT; ++q)
      acc += cv[r * NVOIGT + q] * eps[q];
    sig[r] = g * acc;
  }
}

// the Voigt row of stress (i, k): normal on the diagonal, else its shear pair
__device__ __forceinline__ int voigt_row(const int i, const int k) {
  return i == k ? i : PAIR_ROW(min(i, k), max(i, k));
}

// the cell of the slab thread at node A, on the wall side, or false
__device__ __forceinline__ bool own_cell(const int *A, const int e,
                                         const int lo, const int *NN, int *X) {
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    X[d] = A[d];
    if (d != e && A[d] > NN[d] - 3)
      return false;
  }
  X[e] -= A[e] > lo;
  return true;
}

// the cell around node A that has it at corner m, or false outside
__device__ __forceinline__ bool around(const int s, const int *A, const int *NN,
                                       int *X, int &m) {
  m = (1 << NDIM) - 1 - s;
  bool inside = true;
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    X[d] = A[d] - 1 + ((s >> d) & 1);
    inside &= X[d] >= 1 && X[d] <= NN[d] - 3;
  }
  return inside;
}

// row d of the stretched gradient: the forward strain memory of axis d
__device__ __forceinline__ void strain_memory(const int *X, const int d,
                                              const Layer &l, const int *NN,
                                              const int *S, real_t *p) {
  const int c = slab_at(X, true, d, l, NN, S);
#pragma unroll
  for (int j = 0; j < NDIM; ++j)
    p[j] = c < 0 ? (real_t)0 : l.aux[j * l.size + c];
}

// row d of the adjoint's: the derivative of the node memory through a_node
__device__ __forceinline__ void force_memory(const int *X, const int d,
                                             const Layer &l, const int *NN,
                                             const int *S, const real_t *F,
                                             real_t *p) {
#pragma unroll
  for (int j = 0; j < NDIM; ++j)
    p[j] = (real_t)0;
#pragma unroll
  for (int m = 0; m < (1 << NDIM); ++m) {
    int Y[NDIM];
#pragma unroll
    for (int k = 0; k < NDIM; ++k)
      Y[k] = X[k] + ((m >> k) & 1);
    const int c = slab_at(Y, false, d, l, NN, S);
    if (c < 0)
      continue;
    const real_t w = corner_weight(m, d, F) * l.profile[Y[d]];
#pragma unroll
    for (int j = 0; j < NDIM; ++j)
      p[j] += w * l.aux[(NDIM + j) * l.size + c];
  }
}

// the force of the memory p in row e of a cell's gradient, at its corner m
__device__ __forceinline__ void memory_force(const real_t *p, const int e,
                                             const int m, const real_t g,
                                             const real_t *__restrict__ cv,
                                             const real_t *F, real_t *out) {
  real_t D[NDIM][NDIM], sig[NVOIGT];
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
#pragma unroll
    for (int j = 0; j < NDIM; ++j)
      D[k][j] = k == e ? p[j] : (real_t)0;
  cell_stress(D, cv, g, sig);
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
#pragma unroll
    for (int k = 0; k < NDIM; ++k)
      out[i] -= corner_weight(m, k, F) * sig[voigt_row(i, k)];
}

#if NDIM == 1
#define LAYERS const Layer L[1] = {layer_of(0, aux, profile, layers, NN, S)}
#elif NDIM == 2
#define LAYERS                                                                 \
  const Layer L[2] = {layer_of(0, aux, profile, layers, NN, S),                \
                      layer_of(1, aux, profile, layers, NN, S)}
#elif NDIM == 3
#define LAYERS                                                                 \
  const Layer L[3] = {layer_of(0, aux, profile, layers, NN, S),                \
                      layer_of(1, aux, profile, layers, NN, S),                \
                      layer_of(2, aux, profile, layers, NN, S)}
#endif

// the thread's slab node on axis e, its layer and the profile rows there
#define PML_OR_RETURN                                                          \
  PML_AXES;                                                                    \
  LAYERS;                                                                      \
  const int lo = L[e].lo, hi = L[e].hi;                                        \
  SLAB_OR_RETURN(0);                                                           \
  const real_t *__restrict__ pe = L[e].profile;                                \
  real_t *__restrict__ me = L[e].aux

// -------------------------------------- kernels
extern "C" {

// ------------------------------------------------------------------------------------
__global__ void
fd_kernel(const real_t *__restrict__ u0, const real_t *__restrict__ u1,
          real_t *__restrict__ u2, const real_t *__restrict__ minv,
          const real_t *__restrict__ cell, const real_t *__restrict__ stencil,
#ifdef USE_DAMPING
          const real_t *__restrict__ damping, const real_t dt,
#endif
          const int cs, AXIS_PARAMS) {
  INTERIOR_OR_RETURN;
  AXIS_GEOM;

  real_t force[NDIM];
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
    force[i] = (real_t)0;

#pragma unroll
  for (int c = 0; c < CELLS; ++c) {
    int base, radius, self;
    if (!cell_inside(c, A, NN, S, base, radius, self))
      continue;
    const real_t gc = cell[idx + base];
    const real_t *__restrict__ table = stencil + (radius - 1) * NLOC * NLOC;
#pragma unroll
    for (int l = 0; l < CELLS; ++l) {
      int off;
      if (!block_offset(c, l, radius, S, off))
        continue;
#pragma unroll
      for (int j = 0; j < NDIM; ++j) {
        const real_t uj = u1[j * cs + idx + off];
#pragma unroll
        for (int i = 0; i < NDIM; ++i)
          force[i] -=
              gc * table[(self * NDIM + i) * NLOC + l * NDIM + j] * uj;
      }
    }
  }

  const real_t mi = minv[idx];
#ifdef USE_DAMPING
  const real_t beta = 0.5f * mi * damping[idx] * dt;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    const int n = i * cs + idx;
    u2[n] = (2.f * u1[n] - u0[n] * (1.f - beta) + f0 * mi * force[i]) /
            (1.f + beta);
  }
#else
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    const int n = i * cs + idx;
    u2[n] = -u0[n] + 2.f * u1[n] + f0 * mi * force[i];
  }
#endif
}

// ------------------------------------------------------------------------------------
__global__ void pml_cell_kernel(PML_ARGS) {
  PML_OR_RETURN;
  int X[NDIM];
  if (!own_cell(A, e, lo, NN, X))
    return;
  real_t G[NDIM][NDIM];
  cell_gradient(u1, X, cs, S, F, G);
  const real_t ah = pe[2 * NN[e] + X[e]], bh = pe[3 * NN[e] + X[e]];
#pragma unroll
  for (int j = 0; j < NDIM; ++j)
    me[j * size + t] = bh * me[j * size + t] + ah * G[e][j];
}

// ------------------------------------------------------------------------------------
__global__ void pml_force_kernel(PML_ARGS) {
  PML_OR_RETURN;
  real_t force[NDIM], div[NDIM];
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
    force[i] = div[i] = (real_t)0;
#pragma unroll
  for (int s = 0; s < (1 << NDIM); ++s) {
    int X[NDIM], m;
    if (!around(s, A, NN, X, m))
      continue;
    const real_t g = cell[flat(X, S)];
    real_t G[NDIM][NDIM], p[NDIM], sig[NVOIGT];
    cell_gradient(u1, X, cs, S, F, G);
    // the stretched stress carries every lined axis' memory
#pragma unroll
    for (int d = 0; d < NDIM; ++d) {
      strain_memory(X, d, L[d], NN, S, p);
#pragma unroll
      for (int j = 0; j < NDIM; ++j)
        G[d][j] += p[j];
      if (d == e)
        memory_force(p, e, m, g, cv, F, force);
    }
    cell_stress(G, cv, g, sig);
#pragma unroll
    for (int i = 0; i < NDIM; ++i)
      div[i] -= corner_weight(m, e, F) * sig[voigt_row(i, e)];
  }
  const real_t an = pe[a], bn = pe[NN[e] + a], mi = f * minv[idx];
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    real_t *z = me + (NDIM + i) * size + t;
    *z = bn * *z + an * div[i];
    u2[i * cs + idx] += mi * (force[i] + *z);
  }
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_memory_kernel(PML_ARGS) {
  PML_OR_RETURN;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    real_t *z = me + (NDIM + i) * size + t;
    *z = pe[NN[e] + a] * *z + u1[i * cs + idx];
  }
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_cell_kernel(PML_ARGS) {
  PML_OR_RETURN;
  int X[NDIM];
  if (!own_cell(A, e, lo, NN, X))
    return;
  real_t G[NDIM][NDIM], p[NDIM], sig[NVOIGT];
  cell_gradient(u1, X, cs, S, F, G);
  // the node memories pushed back through the stretched stress
#pragma unroll
  for (int d = 0; d < NDIM; ++d) {
    force_memory(X, d, L[d], NN, S, F, p);
#pragma unroll
    for (int j = 0; j < NDIM; ++j)
      G[d][j] += p[j];
  }
  cell_stress(G, cv, cell[flat(X, S)], sig);
  const real_t bh = pe[3 * NN[e] + X[e]];
#pragma unroll
  for (int j = 0; j < NDIM; ++j)
    me[j * size + t] = bh * me[j * size + t] - sig[voigt_row(e, j)];
}

// ------------------------------------------------------------------------------------
__global__ void pml_adjoint_force_kernel(PML_ARGS) {
  PML_OR_RETURN;
  real_t force[NDIM];
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
    force[i] = (real_t)0;
#pragma unroll
  for (int s = 0; s < (1 << NDIM); ++s) {
    int X[NDIM], m;
    if (!around(s, A, NN, X, m))
      continue;
    real_t p[NDIM];
    // the strain memory through a_half, then the node memory's stress
    strain_memory(X, e, L[e], NN, S, p);
    const real_t w = corner_weight(m, e, F) * pe[2 * NN[e] + X[e]];
#pragma unroll
    for (int i = 0; i < NDIM; ++i)
      force[i] += w * p[i];
    force_memory(X, e, L[e], NN, S, F, p);
    memory_force(p, e, m, cell[flat(X, S)], cv, F, force);
  }
  const real_t mi = f * minv[idx];
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
    u2[i * cs + idx] += mi * force[i];
}

} // extern "C"
