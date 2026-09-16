# Maxwell

**PolarizedWave** solves Maxwell's equations in the two out-of-plane reductions that stay scalar, so that transient photonic inverse design runs on the [scalar](scalar.md) kernels rather than on a vector curl-curl of its own

$$\varepsilon\ddot{\mathbf{E}}=-\nabla\times\left(\frac{1}{\mu}\nabla\times\mathbf{E}\right)$$

with the permittivity `permittivity1` / `permittivity2` $\varepsilon$ that the design $\gamma$ interpolates between, and the permeability `permeability` $\mu$, which the media here are non-magnetic in and which is therefore never designed. This is the classical Yee scheme with $\mathbf{H}$ eliminated: the second-order curl-curl form is what the repo's three-term march, its [sensitivity](sensitivity.md) variants and its `stable_timestep` power iteration are all built around

## the two reductions

Assume no variation out of plane. The curl-curl operator then annihilates the in-plane part of whichever polarization is chosen, one component survives, and the remaining equation is a flux divergence rather than a curl of a curl

| class | unknown | $m$ | $k$ | label |
|---|---|---|---|---|
| `ElectricWave` | $E_z$ | $\varepsilon$ | $1/\mu$ | TE in [Christiansen & Sigmund 2021](https://doi.org/10.1364/JOSAB.406048) |
| `MagneticWave` | $H_z$ | $\mu$ | $1/\varepsilon$ | TM in the same tutorial |

$$\textrm{electric:}\qquad\varepsilon\ddot{E}_z=\nabla\cdot\left(\frac{1}{\mu}\nabla E_z\right)\qquad\qquad\textrm{magnetic:}\qquad\mu\ddot{H}_z=\nabla\cdot\left(\frac{1}{\varepsilon}\nabla H_z\right)$$

So both are [PressureWave](scalar.md) with its two coefficient fields exchanged, and nothing below the Python layer is new: the same `scalar.cu` and `scalar_sensitivity.cu` serve them. `ElectricWave` is `AcousticWave` under the change of variables $\rho_1=\rho_2=\mu$ and $\kappa_i=1/\varepsilon_i$, which `test_it_is_the_acoustic_solver_under_the_change_of_variables` pins to the last bit rather than leaving as a claim

The **TE/TM labels are not portable**, so the classes are named for the field they carry: what a waveguide community calls TM is what the photonic topology optimization tutorial calls TE. Pick the class by which component the problem's boundary conditions are written on, not by the label

## members

| member | signature | description |
|---|---|---|
| constructor | `PolarizedWave(Nx, dx, N, dt, threads, permittivity1=None, permittivity2=None, permeability=1.0)` | on top of [Simulation](wave.md), the two phases and the permeability, rejecting a non-positive material and any grid past 2D |
| timestep bound | `light_speed` | $1/\sqrt{\min(\varepsilon)\mu}$, the speed `wave.stable_dt` has to be given |
| interpolation | `permittivity(indicator)` | $\varepsilon_1+\gamma\left(\varepsilon_2-\varepsilon_1\right)$, linear in the design |
| uniform field | `constant(value)` | `value` over the padded grid, for the coefficient the design does not enter |
| per-axis factors | `step_factors()` | $2\Delta t^2/\Delta x_d^2$, both materials already sitting in the fields |
| source scaling | `source_factor()` | $1$ |

| member | `ElectricWave` | `MagneticWave` |
|---|---|---|
| `parametrization(indicator)` | $k=1/\mu$ constant, $m=\varepsilon$ | $k=1/\varepsilon$, $m=\mu$ constant |
| `parametrization_jacobian(indicator)` | $\left(\varepsilon_2-\varepsilon_1,\,0\right)$, both scalars | $\left(0,\,-\left(\varepsilon_2-\varepsilon_1\right)/\varepsilon^2\right)$, the second a field |

The permittivity is interpolated linearly in $\gamma$ following the tutorial, which is what makes `ElectricWave`'s jacobian a constant. `MagneticWave` carries the same linear permittivity, but its stiffness is $1/\varepsilon$, which is not affine in $\gamma$, so its jacobian is nodal

## the timestep

$$\Delta t\le\frac{1}{c_\textrm{max}\sqrt{\sum_d\Delta x_d^{-2}}},\qquad c_\textrm{max}=\frac{1}{\sqrt{\min\left(\varepsilon\right)\mu}}$$

which is what `wave.stable_dt` returns at `space_order` 2, exactly. The speed to hand it is the **vacuum** one: the background is the fast phase in photonics, the opposite of the seismic case where the design is the slow inclusion. Above order 2 the bound is about 1% optimistic, since it is built from the central second-derivative weights rather than the staggered first-derivative ones, so keep a safety factor

## boundary conditions

| condition | `ElectricWave` | `MagneticWave` |
|---|---|---|
| `Dirichlet` | perfect electric conductor, $E_z=0$ | perfect magnetic conductor, $H_z=0$ |
| `Neumann` | perfect magnetic conductor, the default | perfect electric conductor, the default |
| `sponge` | a real conductivity $\sigma$, not an absorbing artifact | the magnetic dual of one |

The damped update is $m\ddot{u}+d\dot{u}=\nabla\cdot\left(k\nabla u\right)$, and for `ElectricWave` that is Maxwell in a conducting medium with $d=\sigma$ exactly, so opening the domain costs no modelling fiction. What it is **not** is a perfectly matched layer: it reflects at oblique incidence, and it terminates a guided waveguide mode considerably worse than it terminates a plane wave. Size it against the guided mode, not the free-space wavelength

`sponge` takes `beta`, the decay **per step**, so the physical conductivity it produces scales as $1/\Delta t$ and a layer tuned at one resolution detunes at another. Set it from the timestep instead, `beta` $=\alpha\Delta t$ with $\alpha$ the decay per wavelength travelled, which is what the [tpto](tpto.md) drivers do; around $4\Delta t$ over two wavelengths reflects some $-28$ dB

## limits

The adjoint is the exact transpose of the discretization at `space_order` 2 and, for `ElectricWave`, at every order: the design enters its inertia, and the inertia density $-\lambda^n\left(u^{n+1}-2u^n+u^{n-1}\right)/\Delta t^2$ carries no stencil for a wide-order closure to spoil. `MagneticWave` puts the design in the **stiffness**, which is where the graded wall closure and the telescoped flux stop being symmetric above order 2, and a permittivity contrast of 12 makes that error large rather than academic. **Run `MagneticWave` at `space_order` 2**; [sensitivity](sensitivity.md) carries the general statement

Order 2 costs nothing in accuracy here. A binary design is a material jump defined at grid resolution, and the harmonic cell mean across a jump is first order whatever the stencil, so the interface sets the error: a quarter-wave silicon slab transmits within 2% of Fabry-Perot at 20 points per silicon wavelength at order 2 and at order 4 alike

Dispersion and loss are not modelled. A Lorentz or Drude pole needs an auxiliary equation per node, which is a new material model rather than a new parametrization, so a single real index has to serve every wavelength a broadband run scores. For silicon over a telecom band that is a fair approximation and over the visible it is not

## MaxwellWave

**MaxwellWave** solves the vector curl-curl equation on the Yee lattice, which is what 3D needs and 2D does not

$$\varepsilon\ddot{\mathbf{E}}+\sigma\dot{\mathbf{E}}=-\nabla\times\left(\frac{1}{\mu}\nabla\times\mathbf{E}\right)$$

Eliminating $\mathbf{H}$ from the classical Yee update recovers exactly this stencil, so nothing is given up by writing the scheme second order in time, and everything the repo builds on a symmetric second-order operator is inherited: the three-term march, all four [sensitivity](sensitivity.md) variants, and `stable_timestep`

### the lattice is the elastic one

| quantity | lives at | the elastic point it shares |
|---|---|---|
| $E_x$ | $\left(i+\tfrac{1}{2},\,j,\,k\right)$ | the displacement $u_x$ |
| $\left(\nabla\times\mathbf{E}\right)_z$, so $H_z$ | $\left(i+\tfrac{1}{2},\,j+\tfrac{1}{2},\,k\right)$ | the shear stress $\sigma_{xy}$ |

So `MaxwellWave` is [ElasticWave](elastic.md) with the normal-stress block deleted, and it reuses that module's whole lattice layer: `component_offsets`, `component_weights`, `point_average` and their adjoints, all promoted into [wave](wave.md). There are $\textrm{ndim}\left(\textrm{ndim}-1\right)/2$ curl components, one per axis pair, so one in 2D and three in 3D

Every Maxwell difference is a **half-point** difference, since each term of $b_p=\partial_kE_l-\partial_lE_k$ differentiates a component along an axis it is not staggered on. The node-centred `rad_node` never appears, and with it go the zero-row closure, the plane-stress condensation and the clamped fold that the elastic normal strains need. At order 2 the curl is therefore full order everywhere including the wall-adjacent points

### the operator

$$\mathbf{L}=-\mathbf{C}^\top\,\textrm{diag}\left(w_p\nu\right)\mathbf{C},\qquad W=\frac{1}{2}\sum_pw_p\,\nu\,b_p^2$$

with the discrete curl $\mathbf{C}$, the inverse permeability $\nu=1/\mu$ on the pair points, and the cell weight $w_p$ halved on the walls of the axes the pair does not span. The kernels are written as the variational derivative of the magnetic energy $W$ rather than from the Levi-Civita symbol, which is why no sign table appears in either of them: each sign enters $W$ twice and squares away

| property | why it holds | what needs it |
|---|---|---|
| exactly symmetric | both kernels key every tap off the pair point it belongs to | the adjoint is the exact transpose, at **every** order |
| negative semidefinite | $W$ is a sum of squares | the leapfrog is stable, with no material to check |
| material never differentiated | $\nu$ and $\varepsilon$ enter as point multipliers | the gradient needs no derivative of the stencil |
| annihilates discrete gradients | $\mathbf{C}\nabla_h=0$ identically | the discrete $\nabla\cdot\left(\varepsilon\mathbf{E}\right)$ is conserved exactly |

$w_p$ is computed inside the kernel rather than stored: it is identically 1 in 2D, and in 3D it is $\tfrac{1}{2}$ on the walls of the one remaining axis. Applying it once, on the pair point, is what leaves the two launches an exact transpose pair whether it comes from a field or from an index

### members

| member | signature | description |
|---|---|---|
| constructor | `MaxwellWave(Nx, dx, N, dt, threads, permeability=1.0)` | on top of [Simulation](wave.md), the background permeability, rejecting 1D and any face that is not `Conductor` or `Magnetic` |
| staggering | `component_offsets` | component `c` half a node up axis `c`, what [distribute](utils.md) shifts by |
| curl components | `npairs` | one per axis pair, 1 in 2D and 3 in 3D |
| strip radius | `reach` | $2r-1$, the nodes one step reads past a point |
| materials | `build_materials(indicator)` | the point inverse permittivity per component, the pair inverse permeability, and `damping` |
| step | `define_step(kernels, mat)` | the two-launch closure: the curl kernel, then the update |
| inertia | `inverse_inertia(indicator)` | nodal $1/\left(\varepsilon W\right)$, what a [sponge](boundary.md) scales its conductivity by |
| per-axis factors | `step_factors()` | $\Delta t^2/\Delta x_d$, the curl carrying the other $1/\Delta x$ |
| source scaling | `source_factor()` | $1$; the excitation weights divide by the volume the kernel inertia leaves out |
| kernel options | `compile_flags` | adds `-DUSE_MAGNETIC` where `magnetic` is set |

`DielectricWave` supplies the parametrization: the permittivity interpolates linearly between `permittivity1` and `permittivity2` and the permeability stays uniform, so its jacobian is $\left(\varepsilon_2-\varepsilon_1,\,0\right)$

`magnetic` is the switch a parametrization that also designs $\mu$ sets, in the way `derive_inertia` marks the scalar equations where $m=k$. It adds the pair-point curl-against-curl density to both adjoint kernels; left unset, which is what every dielectric problem wants, those kernels carry no stencil at all and the second gradient comes back as zeros

### the null space

The discrete curl-curl annihilates every discrete gradient, so the operator has a null space the size of the node count. Those modes are **not** harmless: projecting the semi-discrete system onto one gives

$$\frac{\textrm{d}^2}{\textrm{d}t^2}\left(\mathbf{g}^\top\mathbf{M}\mathbf{E}\right)=-\boldsymbol{\phi}^\top\left(\nabla\cdot\mathbf{f}\right)$$

so the static component is driven by the **divergence of the source** and marches as $z^{n+1}=2z^n-z^{n-1}$, a Jordan block at eigenvalue 1. A source with nonzero time mean therefore drives linear growth, and one with a nonzero first moment leaves a permanent static blob behind

Two rules keep it quiet, and both cost nothing. Drive a current that does not vary along its own polarization axis, so its discrete divergence vanishes identically; and use `ricker` ([signals](signals.md)), whose integral and first moment both vanish, rather than `sineburst`, whose first moment does not. `examples/forward/maxwellND.py` does both

The reverse march of `reconstruction_sensitivity` is stable on the null space in the same sense the [elastic](elastic.md) rigid-body modes are: perturbations grow linearly in the step count, not exponentially. The difference is that there are order-$N_\textrm{nodes}$ of them rather than a handful, so the aggregate drift is larger and worth watching. `info["drift"]` is the instrument, and the [tpto](tpto.md) drivers print it every iteration

### limits

**Subpixel accuracy.** The permittivity is arithmetically averaged onto each component point, which is right for the field component **tangential** to a material interface and wrong for the normal one, where the harmonic mean is the correct effective medium ([Farjadpour et al. 2006](https://doi.org/10.1364/OL.31.002972)). The consequence is first-order convergence at a staircased interface whatever `space_order` says, which caps how closely a grey design and its thresholded twin can agree. The scalar reductions do not share this: $E_z$ and $H_z$ are always tangential

No perfectly matched layer, and no dispersive materials; see the reduction limits above, which apply unchanged

The kernels are documented in [forward CUDA](cuda_maxwell.md) and [backward CUDA](cuda_maxwell_sensitivity.md)
