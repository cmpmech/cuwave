# Stencils

**Stencils** builds the central finite-difference weights of any even order and emits them as the CUDA preamble the kernels are compiled against

The scheme is nodal but its fluxes live on cells, so what the kernels need is not the second-derivative stencil itself but the cell-flux coefficients it telescopes into; see [forward CUDA](cuda_wave.md) for the flux $D_i^+$ they build and how it is applied

| member | signature | description |
|---|---|---|
| weights | `weights(R)` | the central second-derivative weights $w_{-R..R}$ of order $2R$, from a Vandermonde solve against the Taylor conditions |
| coefficients | `cell_coefficients(R)` | the cumulative tail sums of `weights(R)`, which are the coefficients of the cell flux at radius $R$ |
| preamble | `preamble(space_order)` | the `#define STENCIL_RADIUS` and `#define OP_COEFFS` lines, one table row per radius $1..R$ |

$$\sum_{k=-R}^{R}w_k\,u_{i+k}\approx h^2\frac{\partial^2u}{\partial x^2}\bigg|_i,\qquad c_{R,k}=\sum_{j>k}w_{R+j}$$

with the radius `R` $R$, so `space_order` $2R$ is the formal order in the interior and $c_{R,k}$ the coefficient of $u_{i+k}-u_{i-(k-1)}$ in the flux through the cell at $i+\tfrac{1}{2}$

The table carries **every** radius from 1 to $R$, not just $R$ itself, because the kernel grades its stencil down towards a wall: a node one spacing from the boundary is evaluated at radius 1 and only the deep interior reaches $R$. Rows are zero-padded to width $R$ so the table is rectangular and can live in `__constant__` memory, indexed by a radius the whole warp shares

The preamble is prepended to the kernel source rather than passed as an `nvcc -D` flag, because `RawModule` caches on the code string: a different `space_order` is then a different module automatically, without the order having to appear in the compile options
