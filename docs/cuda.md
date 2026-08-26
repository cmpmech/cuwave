# CUDA kernels

## compilation logic
`compile_kernels(sim, path)` builds one `cp.RawModule` per setup: the kernels are written once for any dimension, precision and order, and the specifics arrive as `nvcc` flags or as injected source

| flag                          | set from                                                              | effect                                                        |
| ----------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------- |
| `-DNDIM=1\|2\|3`              | `sim.ndim`                                                            | selects the number of stencil axes                            |
| `-DUSE_FLOAT`                 | `precision == "float32"`                                              | `real_t = float`, otherwise `double`                          |
| `-DUSE_DAMPING`               | `sim.compile_flags`, once `build_materials` was given a damping field | compiles the damping term and its two extra kernel parameters |
| `STENCIL_RADIUS`, `OP_COEFFS` | `stencils.preamble(space_order)`                                      | the radius $R$ and the face-coefficient table                 |
| `--use_fast_math`             | always                                                                |                                                               |

The stencil table is **prepended to the source text** rather than passed as a `-D`, because `RawModule` caches on the code string: a different `space_order` is then a different module automatically, without the order having to appear in the options

`stencils.preamble(4)` for instance emits
```c
#define STENCIL_RADIUS 2
#define OP_COEFFS { {1.0, 0.0}, {1.25, -0.08333333333333333} }
```
which are the two face-flux stencils, written out — note they approximate the derivative on the cell **face** $x_{i+1/2}$, not on the node $x_i$
$$\textrm{row 1 (radius 1, order 2):}\qquad D_i^+=u_{i+1}-u_i$$
$$\textrm{row 2 (radius 2, order 4):}\qquad D_i^+=\frac{5}{4}\left(u_{i+1}-u_i\right)-\frac{1}{12}\left(u_{i+2}-u_{i-1}\right)$$
with $D_i^+/\Delta x\approx\partial u/\partial x$ at $x_{i+1/2}$, and $D_i^-=D_{i-1}^+$ the flux through the opposite face

## stencil helpers
Two identifiers for one table: `OP_COEFFS` is the macro holding the brace initializer that `preamble` generated, `OP_C` is the `__constant__` array on the device it initializes: `__constant__` because every thread at a given radius reads the same entry, and such a warp-uniform read broadcasts from the constant cache in one transaction

| helper | definition | purpose |
|---|---|---|
| `OP_C[R][R]` | `= OP_COEFFS` | the table, row $r$ holding a complete stencil of radius $r$ (order $2r$), zero-padded to width $R$ |
| `OP_W(r, k)` | `OP_C[(r)-1][(k)-1]` | accessor hiding the 1-based → 0-based shift: `r` the radius a node uses, `k` which term of it |
| `CLOSURE(a, N)` | `min(R, min(a, N-1-a))` | the radius a node at index `a` is allowed |

`CLOSURE` grades the radius **down** towards a wall, since `a` and `N-1-a` are the distances to the two ghost nodes: the first interior node gets radius 1 and hence order 2, the deep interior saturates at $R$. There is only one ghost node per side, so a wide stencil would otherwise reach into memory carrying no boundary information — and because each table row is a self-consistent stencil, the wall-adjacent node gets a genuine order-2 formula rather than a truncated high-order one

At $R=1$ both collapse to compile-time constants (`OP_W` $\equiv1$, `CLOSURE` $\equiv1$): no `__constant__` array, no runtime index, the `k`-loop and its guard fold away, and the generic kernel compiles back to the hand-written order-2 kernel

`flux_divergence_axis` evaluates one axis of $\nabla\cdot\left(k\nabla u\right)$ as the difference of the two cell-face fluxes
$$\textrm{return}=f_d\left(D^+g^+-D^-g^-\right),\qquad g^\pm=\frac{k_ik_{i\pm1}}{k_i+k_{i\pm1}}$$
$$D_i^+=\sum_{\ell=1}^{r}c_\ell\left(u_{i+\ell}-u_{i-\ell+1}\right),\qquad D_i^-=\sum_{\ell=1}^{r}c_\ell\left(u_{i+\ell-1}-u_{i-\ell}\right)$$
with $c$ the table row for the radius $r$ the node was granted and $\ell$ the kernel's loop index `k`. The two nodes of the $\ell$-th pair both sit at distance $\left(\ell-\frac{1}{2}\right)\Delta x$ from the face, which is why the indexing looks lopsided about the node while being symmetric about the face

$g^\pm$ is the *halved* harmonic mean of the two adjacent nodal stiffnesses — the missing factor 2 is the one `step_factors` carries, so the effective face coefficient is $2k_ik_{i\pm1}/\left(k_i+k_{i\pm1}\right)$. That is the series average, the one that keeps the flux single-valued across a material jump, and putting $k$ on the face is what lets the scheme avoid ever differentiating it

### what the coefficients are
The collocated second-derivative weights `weights(R)` satisfy $\sum_jw_j=0$, so that stencil is a discrete divergence and can be split into a difference of face fluxes. `face_coefficients(R)` is the split — the discrete antiderivative of $w$
$$c_\ell=\sum_{j=\ell}^{R}w_j\qquad\Longleftrightarrow\qquad c_\ell-c_{\ell+1}=w_\ell,\quad c_{R+1}=0,\quad w_0=-2c_1$$
so that for uniform $k$ the two face fluxes telescope back to the collocated stencil identically
$$D_i^+-D_i^-=\sum_{j=-R}^{R}w_ju_{i+j}$$
For $R=2$ that is $c_1=w_1+w_2=\frac{4}{3}-\frac{1}{12}=\frac{5}{4}$ and $c_2=w_2=-\frac{1}{12}$

The $c_\ell$ are therefore **not** fitted to make the face flux accurate, and it is not: $D^\pm$ alone stays second order at every $R$, and the order $2R$ appears only once the two are differenced, their $O\!\left(\Delta x^2\right)$ errors cancelling

| measured convergence | $R=1$ | $R=2$ | $R=3$ |
|---|---|---|---|
| $D^+/\Delta x$ against $\partial u/\partial x$ at the face | 2.1 | 2.1 | 2.1 |
| $\left(D^+-D^-\right)/\Delta x^2$ against $\partial^2u/\partial x^2$ | 2.0 | 4.0 | 6.0 |

That cancellation is exact only for uniform $k$. With a varying material it no longer lines up, which is the reason the [sensitivity](sensitivity.md) examples all run at order 2

## fd_kernel
**setup.** `grid_block` maps the fastest axis to grid/block `x`, the next to `y`, the next to `z` (hence the reversed `threads`), so each thread owns one node
- interior guard `a > 0 && a < N-1` on every axis: the ghost ring belongs to the boundary kernel, and it also retires the threads that fall in the padding tail, since the grid is sized on `Nx_padded` while `N` is the logical `Nx`
- `idx = a0*s0 + a1*s1 + a2` is the flat index into the padded array, with the strides `Simulation` derived — the last axis has unit stride, so the innermost axis maps to consecutive threads and the accesses coalesce
- `r0, r1, r2 = CLOSURE(a, N)` per axis, so the order is chosen per node

**Laplacian.** the sum of `flux_divergence_axis` over the axes, each with its own factor $f_d$ and radius $r_d$

**time integration.** central difference in time, explicit, with $\Delta t^2$ already folded into $f_d$
$$u^{n+1}=-u^{n-1}+2u^n+m^{-1}\mathcal{L}\left[u^n\right]$$
and with damping — $\dot{u}$ likewise central, which is what makes the scheme stay explicit
$$u^{n+1}=\frac{2u^n-\left(1-\beta\right)u^{n-1}+m^{-1}\mathcal{L}\left[u^n\right]}{1+\beta},\qquad\beta=\frac{d\,\Delta t}{2m}$$
where $m^{-1}$ is `minv[idx]`, or `1 / stiff` where the formulation derives it (see [wave](wave.md))

## bc_kernels

One thread per ghost node in a flat 1D launch, so these kernels index by decoding a linear thread id instead of by the grid mapping above
### homogeneous_neumann_kernel
- the launch covers $\sum_d2\prod_{k\ne d}\left(N_k-2\right)$ nodes: two ghost faces per axis, each holding only the *interior* of the remaining axes, so edges and corners are never written — the stencil reaches across one axis at a time from an interior node and therefore only ever reads face ghosts
- the axis loop subtracts each axis's ghost count from `t` until the thread's own axis is found, then decodes its position within that face by mixed-radix division over the other axes, the `+1` shifting it off the ghost ring
- the write is `u[ghost] = u[mirror]`, i.e. index $0\leftarrow2$ and $N-1\leftarrow N-3$: mirroring makes the central difference across the wall vanish, which is the zero-flux condition. `wave.mirror_ghosts` applies the same rule to the material fields on the host

### dirichlet_kernel
TODO ADD that straight away + all the needed functionality in other files (maybe create the boundary.py file?)

## excitation_kernel
One thread per source, adding the scaled signal sample into the field
- `weight`, $\Delta t^2\,/\,m$ at the source node times `source_factor`, is precomputed once at setup by `excitation_weights`, so this kernel needs neither a material field nor any knowledge of the formulation
- `lin_index` comes from `wave.flatten_indices`, which collapses the `(ndim, num)` grid indices into flat indices of the padded array once, outside the time loop
- `source` is the whole `(N, num_sources)` record with `offset = t * num_sources`, not a row view: slicing a row out per step costs more launch overhead on the host than this kernel takes on the device
- `atomicAdd` rather than `+=`, because nothing stops two entries of `lin_index` from naming the same node — a plain read-modify-write would then silently lose one of them

## get_signal_kernel
The mirror image: one thread per sensor, `um[offset + idx] = u[lin_index[idx]]`, with the same precomputed `lin_index` and the same whole-record-plus-offset convention, so the caller never slices a row either
