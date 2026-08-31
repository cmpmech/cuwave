# Elastic

**ElasticWave** solves the isotropic elastic wave equation for the displacement vector, a sibling of [PressureWave](wave.md) on the same grid, the same three-term time march and the same three [sensitivity](sensitivity.md) variants

$$m\ddot{\mathbf{u}}+d\dot{\mathbf{u}}-\nabla\cdot\boldsymbol{\sigma}=\mathbf{f},\qquad\boldsymbol{\sigma}=\mathbf{C}:\boldsymbol{\varepsilon},\qquad\boldsymbol{\varepsilon}=\frac{1}{2}\left(\nabla\mathbf{u}+\nabla\mathbf{u}^\top\right)$$

with the displacement $\mathbf{u}$ carrying one component per axis, the stiffness tensor $\mathbf{C}$ built from `wavespeed_p` $c_p$, `wavespeed_s` $c_s$ and `density` $\rho_0$, and the damping $d$ the [sponge](boundary.md) sets

## the stencil

The [pressure](wave.md) equation is discretized axis by axis, each axis a flux divergence with the stiffness on the cell between two nodes. That form carries the elastic **axial** terms unchanged, $\partial_x\left(\left(\lambda+2\mu\right)\partial_xu_x\right)$ and $\partial_y\left(\mu\,\partial_yu_x\right)$, but it cannot carry the **cross** terms $\partial_x\left(\lambda\,\partial_yu_y\right)$ and $\partial_y\left(\mu\,\partial_xu_y\right)$: a per-axis flux varies one index at a time, and those need two. So the gather is over cells rather than axes, and the stencil is 9-point in 2D and 27-point in 3D

$$f_x=\left(\lambda+2\mu\right)\left[M_y\otimes D^2_x\right]u_x+\mu\left[M_x\otimes D^2_y\right]u_x+\left(\lambda+\mu\right)\left[D_x\otimes D_y\right]u_y$$

at `space_order` 2, with the second difference $D^2=\left(1,-2,1\right)$, the centred first difference $D=\left(-\tfrac{1}{2},0,\tfrac{1}{2}\right)$ and the transverse average $M=\left(1,4,1\right)/6$. The cross term is the textbook one; the transverse average on the axial terms is the only departure from it, and it is not cosmetic: a plain $\left(1,2,1\right)/4$ leaves the checkerboard $\left(-1\right)^{i+j}$ exactly in the null space, an hourglass mode a material contrast excites and nothing damps

The coefficients are not written by hand. They are derived cell by cell as

$$\mathbf{L}=-\sum_\textrm{cells}\gamma_c\,\mathbf{B}^\top\mathbf{C}\,\mathbf{B},\qquad\gamma_c=\frac{2^\textrm{ndim}}{\sum_\textrm{nodes}\gamma_i^{-1}}$$

with $\mathbf{B}$ the discrete symmetric gradient over the cell's $2^\textrm{ndim}$ nodes, $\gamma_c$ the harmonic mean of the design field over them, and the sum running over the cells **inside** the domain only. Deriving them that way rather than writing the stencil out is what buys four properties at once, and every one of them is what a later stage rests on:

| property | why it holds | what needs it |
|---|---|---|
| exactly symmetric | $\mathbf{B}^\top\mathbf{C}\mathbf{B}$ is symmetric for any $\mathbf{B}$ and symmetric $\mathbf{C}$, so a **varying** material needs no check | the adjoint is the exact transpose |
| negative semidefinite | $\mathbf{C}$ is positive definite | the leapfrog is stable, with no material to check |
| material never differentiated | $\mathbf{C}$ enters as a cell multiplier | the gradient needs no derivative of the stencil |
| traction free without a kernel | restricting the sum to interior cells makes $\boldsymbol{\sigma}\cdot\mathbf{n}=0$ the natural condition | a free surface on faces, edges and corners alike |

Symmetry is the one worth dwelling on, because it is what buys the higher orders. The pressure stencil pairs a **wide** inner gradient against a **single** outer difference, which is why it stops being the exact transpose above order 2 (see [sensitivity](sensitivity.md)). Here both sides of the sandwich widen together, so the transpose stays exact at any width

## higher order

`space_order` is $2r$ for a stencil radius $r$, and any even order is allowed. The cell reaches $2r$ nodes per axis instead of 2, its strain built from Lagrange weights through them, so the nodal stencil grows to $\left(4r-1\right)^\textrm{ndim}$ points. Two things keep it well behaved:

| ingredient | why |
|---|---|
| enough quadrature points, $\max\left(2,\,2r-1\right)$ per axis | the product $\mathbf{B}^\top\mathbf{C}\mathbf{B}$ has to be integrated exactly for the order to show, and never fewer than two, or the checkerboard returns |
| a graded wall closure | a cell nearer a wall than $r$ drops to the radius its distance allows, so the stencil never reaches a ghost node and the interior-cell sum still gives a free surface |

Grading costs accuracy in a layer a few nodes deep and costs symmetry nothing: the sandwich is symmetric whatever $\mathbf{B}$ each cell carries, so a cell at reduced radius is as exact a transpose as one in the deep interior

The price is arithmetic. A node gathers $\left(2r\right)^\textrm{ndim}$ cells and each carries $\left(2r\right)^\textrm{ndim}$ nodes, so the work per node runs as $\left(2r\right)^{2\,\textrm{ndim}}$

| `space_order` | stencil (2D) | cost per step, 2D |
|---|---|---|
| 2 | 9 | 1x |
| 4 | 49 | 15x |
| 6 | 121 | 76x |

Against that a higher order resolves a wavelength on fewer points, and a coarser grid buys both fewer nodes and a larger timestep, so the comparison turns on **what sets the spacing**

| what limits the grid | can the grid coarsen? | order 4 against order 2, 2D |
|---|---|---|
| dispersion, a smooth medium over a long path | yes, to about a third of the spacing | about $0.35$x, so a win of roughly 3 |
| geometry, a defect a few nodes across | no | $14.6$x, a pure loss |

The second row is the one an [fwi](fwi.md) driver sits in: the grid is set by the smallest defect and not by the wavelength, so the spacing cannot follow the order and the extra accuracy buys nothing. Folding the quadrature into a stored stress field would trade memory for a factor of a few on the per-step figure, and is the obvious next move if high order turns out to matter

There is a second and sharper price, and it decides where a wide stencil is usable at all. A cell shields a light node from its neighbours through the harmonic mean, but only over the `2**ndim` nodes it owns; a wide stencil reaches past that, so a node at $\gamma=10^{-4}$ feels the full stiffness of solid cells two nodes away and the stable timestep collapses with it

| `space_order` | uniform | $\gamma=10^{-1}$ void | $\gamma=10^{-4}$ void |
|---|---|---|---|
| 2 | 1.48 | 1.44 | 1.39 |
| 4 | 1.48 | 1.30 | **0.19** |
| 6 | 1.30 | 1.16 | **0.12** |

as a multiple of `wave.stable_dt`. The cliff sits between $10^{-2}$ and $10^{-3}$: above it a wide stencil costs a few percent of timestep and nothing else, and a design floor at $\gamma=0.1$, which is where an ultrasonic inversion bounds the density anyway, pays $1.11$x. Below it the timestep collapses and a higher order stops being affordable. The scheme stays correct at the measured timestep either way, so this is a cost question and not a stability bug

`stable_timestep(sim, indicator)` is what tells the two apart. It power-iterates the step kernel itself, so it is exact for any order, material and boundary layout, where `wave.stable_dt` knows only the speed and the spacing. A number far below it is the signal to drop `space_order`, not to shrink `dt`

## members

| member | signature | description |
|---|---|---|
| constructor | `ElasticWave(Nx, dx, N, dt, threads, density=None, wavespeed_p=None, wavespeed_s=None, plane="strain")` | on top of [Simulation](wave.md), the two speeds and the background density, rejecting a shear speed above the pressure one |
| stencil radius | `radius` | half `space_order`, the nodes the cell reaches per axis being twice it |
| timestep | `stable_timestep(sim, indicator, iterations=60, safety=0.95)` | the largest stable step, measured on the step kernel rather than estimated |
| materials | `build_materials(indicator)` | the lumped inverse inertia, the harmonic cell field, the element table and `damping` |
| inertia | `inverse_inertia(indicator)` | $1/\left(\gamma\rho_0V W\right)$, the lumped mass the interior-cell assembly implies |
| coefficients | `parametrization_jacobian(indicator)` | $\left(1,\,1\right)$, since $\gamma$ scales inertia and stiffness alike |
| per-axis factors | `step_factors()` | $\Delta t^2$, the grid spacing already sitting in the element |
| source scaling | `source_factor()` | $1$, since $\rho_0$ is already folded into the lumped inertia |
| stencil | `stencil()` | the table at $\gamma=1$, one graded radius per slice, zero padded and flattened |
| adjoint scaling | `adjoint_weights(sensors)` | ones, since the lumped inertia already carries the cell weights $W$ |
| stiffness matrix | `voigt(ndim, lame, shear, plane="strain")` | the isotropic Voigt matrix, `plane` selecting strain or stress in 2D |
| stencil | `cell_stencil(ndim, dx, C, radius=1)` | one cell's contribution to the nodal stencil, ordered (node, component) |
| block order | `block_indices(ndim, radius)` | the cell's node block in the kernel's order, axis 0 running fastest |
| weights | `lagrange_weights(radius, t)` | value and derivative weights at `t` inside the cell |

## parametrization

A single indicator scales the density while both wave speeds stay fixed, so $\mathbf{C}=\gamma\rho_0\tilde{\mathbf{C}}$ scales inertia and stiffness alike and the stable timestep does not move as the design does

$$\tilde{C}_{ijkl}=\left(c_p^2-2c_s^2\right)\delta_{ij}\delta_{kl}+c_s^2\delta_{ik}\delta_{jl}+c_s^2\delta_{il}\delta_{jk}$$

with the Lamé parameters recovered as `lame` $\lambda=\rho_0\left(c_p^2-2c_s^2\right)$ and `shear` $\mu=\rho_0c_s^2$. This is the parametrization ultrasonic full waveform inversion reaches for, following [Bürchner et al. 2025](https://doi.org/10.1016/j.ultras.2025.107705), since a void is a density contrast at unchanged speeds and the optimization stays on one field

`plane` is a host-side choice and costs the kernel nothing: plane strain takes $\lambda$ as it is, plane stress replaces it by $2\lambda\mu/\left(\lambda+2\mu\right)$, and 3D rejects anything but strain

## boundary conditions

| condition | meaning | cost |
|---|---|---|
| `Traction` | $\boldsymbol{\sigma}\cdot\mathbf{n}=0$, the default and a genuine free surface | no kernel and no launch |
| `Clamped` | $\mathbf{u}=0$, the wall nodes held by a zeroed inverse inertia | no kernel and no launch |
| `sponge` | the same damping field the pressure equation takes, shared by every component | as [boundary](boundary.md) |

A sponge behind an elastic wall has to swallow the shear and surface waves as well as the pressure one, and the shear wavelength is $c_p/c_s$ shorter at the same frequency, so a layer sized on the pressure wavelength leaks. Size it on $c_s$ and expect what is left to show in the tangential component

## limits

Where the pressure equation trades exactness for width above order 2, this one trades cost for it instead and keeps the exact transpose at every order

In 1D there is no coupling, so `ElasticWave` reduces to the [scalar](wave.md) equation with $c=\sqrt{\left(\lambda+2\mu\right)/\rho}$ and reproduces `ScalarWave` node for node. That is worth keeping in reach: it is the one configuration where the whole vector pipeline can be checked against an already-trusted scalar one

The kernels are documented in [forward CUDA](cuda_elastic.md) and [backward CUDA](cuda_elastic_sensitivity.md)
