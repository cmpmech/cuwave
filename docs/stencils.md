# Stencils

**Stencils** builds the central finite-difference weights of any even order and emits them as the CUDA preamble the kernels are compiled against

The scheme is nodal but its fluxes live on cells, so what the kernels need is not the second-derivative stencil itself but the cell-flux coefficients it telescopes into; see [forward CUDA](cuda_scalar.md) for the flux $D_i^+$ they build and how it is applied

| member | signature | description |
|---|---|---|
| weights | `weights(R)` | the central second-derivative weights $w_{-R..R}$ of order $2R$, from a Vandermonde solve against the Taylor conditions |
| coefficients | `cell_coefficients(R, stencil="taylor")` | the cumulative tail sums of `weights(R)`, which are the coefficients of the cell flux at radius $R$, or with `"optimized"` the dispersion-fitted ones |
| coefficients | `staggered_weights(R, stencil="taylor")` | the first-derivative weights $c_{1..R}$ of order $2R$ on the half offsets, what the [elastic](elastic.md) differences tap, fitted the same way with `"optimized"` |
| preamble | `preamble(space_order, stencil="taylor")` | the `#define STENCIL_RADIUS`, `#define OP_COEFFS` and `#define STAG_COEFFS` lines, one table row per radius $1..R$, the full-radius row fitted with `"optimized"` |
| tolerance | `DISPERSION_TOL` | $10^{-3}$, the relative phase velocity error an optimized row keeps over its band |

$$\sum_{k=-R}^{R}w_k\,u_{i+k}\approx h^2\frac{\partial^2u}{\partial x^2}\bigg|_i,\qquad c_{R,k}=\sum_{j>k}w_{R+j},\qquad\sum_{k=1}^{R}\tilde{c}_k\left(u_{+\frac{2k-1}{2}}-u_{-\frac{2k-1}{2}}\right)\approx h\frac{\partial u}{\partial x}\bigg|_0$$

with the radius `R` $R$, so `space_order` $2R$ is the formal order in the interior, $c_{R,k}$ the coefficient of $u_{i+k}-u_{i-(k-1)}$ in the flux through the cell at $i+\tfrac{1}{2}$, and $\tilde{c}_k$ the staggered weights whose antisymmetric taps leave only the $R$ odd Taylor conditions to solve

The table carries **every** radius from 1 to $R$, not just $R$ itself, because the kernel grades its stencil down towards a wall: a node one spacing from the boundary is evaluated at radius 1 and only the deep interior reaches $R$. Rows are zero-padded to width $R$ so the table is rectangular and can live in `__constant__` memory, indexed by a radius the whole warp shares

The preamble is prepended to the kernel source rather than passed as an `nvcc -D` flag, because `RawModule` caches on the code string: a different `space_order` is then a different module automatically, without the order having to appear in the compile options

## optimized rows

`Simulation.stencil = "optimized"` replaces the full-radius row of both tables with weights fitted over a wavenumber band rather than at $kh\to0$, the least-squares fit of [Liu 2013](https://doi.org/10.1190/geo2012-0480.1) for the cell flux and of [Liu 2014](https://doi.org/10.1093/gji/ggu032) for the staggered difference. Each row's symbol over its exact one is linear in the weights,
$$\frac{4\sin\left(\theta/2\right)}{\theta^2}\sum_{k=1}^{R}c_k\sin\left(\left(k-\tfrac12\right)\theta\right)\qquad\textrm{and}\qquad\frac{2}{\theta}\sum_{k=1}^{R}\tilde{c}_k\sin\left(\left(k-\tfrac12\right)\theta\right)$$
with $\theta=kh$, so the fit is one constrained solve per trial band: least squares against 1 over $\left[0,\kappa\right]$ under the consistency condition $\sum_kc_k\left(2k-1\right)=1$, and the widest $\kappa$ whose phase velocity error stays under `DISPERSION_TOL` is kept. The row loses its formal order above 2 but keeps the error under the tolerance to about 1.4 times the wavenumber the Taylor row does, which is what a coarser grid buys back as memory and time

The graded wall rows stay Taylor and `space_order` 2 has nothing to fit, so `"optimized"` changes only the deep interior at orders 4 and up. Every adjoint kernel reads the same tables, so the exact transpose is untouched. The fitted symbol peaks higher at the Nyquist mode $\theta=\pi$, so `stable_dt(dx, wavespeed, space_order, stencil)` evaluates the alternating sum of the row it is given and the stable step shrinks by up to 7%

The fit is spatial only, while the leapfrog's own time error grows as $\left(\omega\Delta t\right)^2$ and has the opposite sign. On a grid coarse enough for the band to matter, the time error at the stable step is the larger one, so an optimized stencil pays only together with a `dt` well below `stable_dt`, the further below the tighter the error target. A binary design is first order at its interfaces whatever the row, so the gain is for smooth media such as a [fwi](fwi.md) model

