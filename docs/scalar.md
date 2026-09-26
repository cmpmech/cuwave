# Scalar

**PressureWave** solves the scalar pressure wave equation for one unknown per node, the cheapest of the three [wave](wave.md) families and the one every driver reaches for first, with `ScalarWave` and `AcousticWave` supplying the two parametrizations of its coefficient fields

$$m\ddot{u}+d\dot{u}-\nabla\cdot\left(k\nabla u\right)=f$$

with the inertia $m$, the damping $d$ and the stiffness $k$, stored as the nodal fields `minv` $=1/m$, `damping` and `stiff` $=k$, since the kernel only ever needs the inverse inertia

One unknown per node and a per-axis flux, so the operator needs no `component_offsets` and no cell gather: the vector siblings [elastic](elastic.md) and [anisotropic](anisotropic.md) pay for the cross terms this equation does not have

`damping` is a constructor field and not a per-call argument, so every path that takes a `Simulation` (`simulate`, `sensitivity` and the [utils](utils.md) glue over them) steps the same operator and no two call sites can disagree about it. `None` is the lossless default, and is what `superposition_sensitivity` requires

`domain` is the other constructor field of the same kind: a nodal mask of the physical domain, for a solid voxelized into a box whose remainder is only there because the grid is structured. Each cell of the flux divergence then takes the radius of the run of domain nodes on either side of it

$$r_{i+\frac{1}{2}}=\min\left(r_\textrm{wall},\,1+n^-_i,\,n^+_i\right)$$

with `CLOSURE` $r_\textrm{wall}$, and $n^\pm_i$ the domain nodes in a row past node $i$ each way along the axis, counted up to $R$. A cell leaving the domain gets radius 0, so it carries no flux at all and the wall is an exact homogeneous Neumann condition on the staircase, and no stencil tap ever reads a node outside. That is what makes the outside free to skip: `fd_kernel` launches only the tiles (one thread block each) holding a domain node, bit for bit the result of launching all of them, and whatever `indicator` holds outside is never read, so no void value $\gamma_\textrm{void}$ is needed

The void a driver otherwise sets outside is a low-impedance medium, not a wall. At order 2 it converges to this one as $\gamma_\textrm{void}\to0$, but a wider stencil reads the void's own wave field with the full cell stiffness, so there the solid depends on a field that means nothing and a frozen outside would be wrong; the domain closure removes that dependence rather than approximating it. `domain_cells` packs the two radii per axis into one int per node, `DEEP` wherever they are all the full ones, and `domain_tiles` marks the tiles holding any other node, the only ones whose threads read it. The price is the setup, a few elementwise passes over the grid per call, and the material gradients: `sensitivity` and its two siblings refuse a domain, since `gradient_kernel` reads every cell at the full radius. `source_sensitivity` needs no gradient kernel and takes one. What it buys scales with the box the domain wastes: little for a disk, most for a scanned solid

`dirichlet` is the homogeneous Dirichlet counterpart, a nodal mask of nodes held at zero, taken out of the stepped set like the outside of `domain` but with the cell onto each of them left **open**: its flux is $k_i\left(0-u_i\right)/\left(h/2\right)$, the zero put at the cell midpoint, which is where the closed Neumann cell puts its wall too, and which a staircase needs, since the true boundary sits on average half a cell past the last stepped node (putting the zero on the held node itself biases the domain outward by that half cell, and the eigenfrequencies with it). It is the limit of the older approach of an infinitely stiff material outside, without its dependence on how stiff. Nothing across is read, neither its $u$ nor its stiffness, so held nodes skip their tiles exactly like the outside of `domain`. The two compose: `domain` alone gives Neumann walls, `dirichlet` alone gives sound-soft obstacles or pinned regions inside the box, and together each wall takes whichever the node across is. A held row inside the box is then the discrete method of images about its cell midpoint, to round-off at order 2. At wider orders the cells next to it grade down to radius 1, as at a Neumann wall, so a grid-aligned wall no longer matches the odd mirror of the full stencil: on a staircase that costs nothing the staircase has not already cost, but it is why order 2 is the sensible choice for masked problems. Inhomogeneous values would need a field for them and are not implemented

`derive_inertia` marks the parametrizations in which $m=k$, so that the fields carry the impedance only and the wave speed sits in `step_factors`: `minv` is then recovered from `stiff` in the kernel, saving one field pass per step and one grid field of memory

The hooks a parametrization fills in are the ones [wave](wave.md) lists for every specialization; the two below differ only in what they put into them

## ScalarWave
$$\gamma\ddot{u}+d\dot{u}-c_0^2\nabla\cdot\left(\gamma\nabla u\right)=\frac{f}{\rho_0}$$
with the background wave speed `wavespeed` $c_0$ and density `density` $\rho_0$, where $\gamma$ scales the density $\rho=\gamma\rho_0$ and hence the impedance, leaving the wave speed at $c_0$ everywhere

| member | specific to `ScalarWave` |
|---|---|
| `parametrization(indicator)` | $k=\gamma$ and $m$ derived, as $\gamma$ scales inertia and stiffness alike (`derive_inertia`) |
| `parametrization_jacobian(indicator)` | $\left(1,\,1\right)$ |
| `step_factors()` | $2\,c_0^2\Delta t^2/\Delta x_k^2$, carrying the wave speed |
| `source_factor()` | $1/\rho_0$ |

## AcousticWave
$$\frac{1}{\kappa}\ddot{u}+d\dot{u}-\nabla\cdot\left(\frac{1}{\rho}\nabla u\right)=f$$
$$\frac{1}{\rho\left(\gamma\right)}=\frac{1}{\rho_1}+\gamma\left(\frac{1}{\rho_2}-\frac{1}{\rho_1}\right),\qquad\frac{1}{\kappa\left(\gamma\right)}=\frac{1}{\kappa_1}+\gamma\left(\frac{1}{\kappa_2}-\frac{1}{\kappa_1}\right)$$
with the two materials `rho1`/`kappa1` and `rho2`/`kappa2` that $\gamma$ interpolates between, $\gamma=0$ giving the first and $\gamma=1$ the second, and the wave speed $c=\sqrt{\kappa/\rho}$ following from the two fields

| member | specific to `AcousticWave` |
|---|---|
| `parametrization(indicator)` | $m=1/\kappa$ and $k=1/\rho$, both stored as the inverse the kernel wants, so neither $\rho$ nor $\kappa$ is ever formed |
| `parametrization_jacobian(indicator)` | $\left(1/\kappa_2-1/\kappa_1,\,1/\rho_2-1/\rho_1\right)$, constant since both coefficients are affine in $\gamma$ |
| `step_factors()` | $2\,\Delta t^2/\Delta x_k^2$, the wave speed already sitting in the fields |
| `source_factor()` | $1$ |

