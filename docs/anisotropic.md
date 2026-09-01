# Anisotropic

**AnisotropicElasticWave** solves the elastic wave equation with the displacement components collocated on the nodes, the order-2 sibling the staggered [ElasticWave](elastic.md) does not replace: it carries any symmetric Voigt matrix `C`, where a staggered grid can only carry couplings between strains that share a point

The governing equation and the $\gamma$ parametrization are those of [elastic](elastic.md); only the discretization differs

## the stencil

The stencil is a 9-point one in 2D and 27-point in 3D, gathered cell by cell rather than axis by axis, since a variable-coefficient elastic operator couples the components through mixed derivatives that a per-axis flux cannot carry when the components share the nodes

$$f_x=\left(\lambda+2\mu\right)\left[M_y\otimes D^2_x\right]u_x+\mu\left[M_x\otimes D^2_y\right]u_x+\left(\lambda+\mu\right)\left[D_x\otimes D_y\right]u_y$$

with the second difference $D^2=\left(1,-2,1\right)$, the centred first difference $D=\left(-\tfrac{1}{2},0,\tfrac{1}{2}\right)$ and the transverse average $M=\left(1,4,1\right)/6$. The cross term is the textbook one; the transverse average on the axial terms is the only departure from it, and it is not cosmetic: a plain $\left(1,2,1\right)/4$ leaves the checkerboard $\left(-1\right)^{i+j}$ exactly in the null space, an hourglass mode a material contrast excites and nothing damps

The coefficients are not written by hand. They are derived cell by cell as

$$\mathbf{L}=-\sum_\textrm{cells}\gamma_c\,\mathbf{B}^\top\mathbf{C}\,\mathbf{B},\qquad\gamma_c=\frac{2^\textrm{ndim}}{\sum_\textrm{nodes}\gamma_i^{-1}}$$

with $\mathbf{B}$ the discrete symmetric gradient over the cell's $2^\textrm{ndim}$ nodes, integrated with two Gauss points per axis (which is what keeps the checkerboard out), $\gamma_c$ the harmonic mean of the design field over the corners, and the sum running over the cells **inside** the domain only. Deriving them that way buys four properties at once

| property | why it holds | what needs it |
|---|---|---|
| exactly symmetric | $\mathbf{B}^\top\mathbf{C}\mathbf{B}$ is symmetric for any $\mathbf{B}$ and symmetric $\mathbf{C}$, so a **varying** material needs no check | the adjoint is the exact transpose |
| negative semidefinite | $\mathbf{C}$ is positive definite | the leapfrog is stable, with no material to check |
| material never differentiated | $\mathbf{C}$ enters as a cell multiplier | the gradient needs no derivative of the stencil |
| traction free without a kernel | restricting the sum to interior cells makes $\boldsymbol{\sigma}\cdot\mathbf{n}=0$ the natural condition | a free surface on faces, edges and corners alike |

## order 2 only

A node gathers $2^\textrm{ndim}$ cells and each cell couples $2^\textrm{ndim}$ nodes, so widening the cell to radius $r$ runs the work per node as $\left(2r\right)^{2\,\textrm{ndim}}$: 63x from order 2 to 4 in 3D, and three orders of magnitude by order 6. That is why the constructor rejects `space_order` above 2 and the staggered [ElasticWave](elastic.md), whose cost is linear in $r$, is the high-order scheme. What this one keeps in exchange is the collocation: every strain component exists at every quadrature point, so `C` may couple all of them, which is general anisotropy

## members

On top of [Simulation](wave.md) and mirroring [ElasticWave](elastic.md), with these its own:

| member | signature | description |
|---|---|---|
| constructor | `AnisotropicElasticWave(Nx, dx, N, dt, threads, density=None, wavespeed_p=None, wavespeed_s=None, plane="strain", C=None)` | the two speeds and the background density; `C` overrides the isotropic Voigt matrix with any symmetric one at $\gamma=1$ |
| stencil | `stencil()` | one cell's $-\mathbf{B}^\top\mathbf{C}\mathbf{B}$ at $\gamma=1$, flattened for the kernel and built once |
| stencil | `cell_stencil(ndim, dx, C)` | the same matrix as a host function, ordered (node, component) in `corner_bits` order |
| averaging | `cell_average(sim, field)` | the harmonic corner mean $\gamma_c$, stored at the cell's low corner |
| inertia | `inverse_inertia(indicator)` | lumped $1/\left(\gamma\rho_0VW\right)$, the mass the interior-cell assembly implies |
| weights | `cell_weights()` | the nodal cell weights $W$, halved once per wall the node sits on |

## boundary conditions

| condition | meaning | cost |
|---|---|---|
| `Traction` | $\boldsymbol{\sigma}\cdot\mathbf{n}=0$, the default and a genuine free surface | no kernel and no launch |
| `Clamped` | $\mathbf{u}=0$, the wall nodes held by a zeroed inverse inertia | no kernel and no launch |
| `sponge` | the same damping field the pressure equation takes | as [boundary](boundary.md) |

In 1D there is no coupling, so `AnisotropicElasticWave` reduces to the [scalar](scalar.md) equation with $c=\sqrt{\left(\lambda+2\mu\right)/\rho}$ and reproduces `ScalarWave` node for node

The kernels are documented in [forward CUDA](cuda_anisotropic.md) and [backward CUDA](cuda_anisotropic_sensitivity.md)
