# Elastic

**ElasticWave** solves the isotropic elastic wave equation for the displacement vector on a staggered grid, a sibling of [PressureWave](scalar.md) on the same three-term time march and the same [sensitivity](sensitivity.md) variants

$$m\ddot{\mathbf{u}}+d\dot{\mathbf{u}}-\nabla\cdot\boldsymbol{\sigma}=\mathbf{f},\qquad\boldsymbol{\sigma}=\mathbf{C}:\boldsymbol{\varepsilon},\qquad\boldsymbol{\varepsilon}=\frac{1}{2}\left(\nabla\mathbf{u}+\nabla\mathbf{u}^\top\right)$$

with the displacement $\mathbf{u}$ carrying one component per axis, the stiffness tensor $\mathbf{C}$ built from `wavespeed_p` $c_p$, `wavespeed_s` $c_s$ and `density` $\rho_0$, and the damping $d$ the [sponge](boundary.md) sets

## the staggering

The [pressure](scalar.md) equation is discretized axis by axis, each axis a flux divergence with the stiffness on the cell between two nodes. That form cannot carry the elastic **cross** terms $\partial_x\left(\lambda\,\partial_yu_y\right)$: a per-axis flux varies one index at a time, and those need two. Placing component `c` half a node up its own axis dissolves the problem, in the staggered layout going back to [Virieux 1986](https://doi.org/10.1190/1.1442147) and widened to fourth order by [Levander 1988](https://doi.org/10.1190/1.1442422): every strain then lands on a natural point through a pure per-axis staggered difference

| quantity | lives at | from |
|---|---|---|
| $u_x$ | $\left(i+\tfrac{1}{2},\,j,\,k\right)$ | the unknown itself |
| $\varepsilon_{xx},\,\varepsilon_{yy},\,\varepsilon_{zz}$ and the normal stresses | the node | $\partial_xu_x$ staggered along $x$ lands back on integers |
| $\varepsilon_{xy}$ and $\sigma_{xy}$ | $\left(i+\tfrac{1}{2},\,j+\tfrac{1}{2},\,k\right)$ | both of its derivatives land there without averaging |

So the cross terms cost no transverse averaging and no cell gather, and the work per node stays **linear** in the stencil radius, where the cell assembly of [AnisotropicElasticWave](anisotropic.md) pays $\left(2r\right)^{2\,\textrm{ndim}}$. The layout reaches the rest of the code through `component_offsets`, which [distribute](utils.md) reads so a source or receiver coordinate interpolates onto each component's own shifted grid

## the operator

The coefficients are not written by hand. The operator is assembled as

$$\mathbf{L}=-\sum_s w_s\,\gamma_s\,\mathbf{B}_s^\top\mathbf{C}\,\mathbf{B}_s$$

with $\mathbf{B}_s$ the staggered symmetric gradient onto stress point $s$, $w_s$ the cell weight (halved per wall the point sits on), and $\gamma_s$ the design field sampled there: the nodal value for the normal family, the harmonic mean of the four straddled nodes for a shear point (`pair_average`), following the volume averaging of [Moczo et al. 2002](https://doi.org/10.1785/0120010167). The inertia is the arithmetic two-node mean at each component's own point (`point_average`). Deriving the operator this way buys the same four properties the cell assembly had, and every one of them is what a later stage rests on

| property | why it holds | what needs it |
|---|---|---|
| exactly symmetric | both kernels key every tap off the stress point it belongs to, so forward and transpose read the same matrix | the adjoint is the exact transpose, at **every** order |
| negative semidefinite | $\mathbf{C}$ is positive definite | the leapfrog is stable, with no material to check |
| material never differentiated | $\gamma_s$ enters as a point multiplier | the gradient needs no derivative of the stencil |
| traction free without a kernel | a wall stress point grades to a zero row and its axis is condensed out | a free surface on faces, edges and corners alike |

Unlike a centred difference, a staggered one does not annihilate the checkerboard $\left(-1\right)^{i+j}$, so the hourglass mode the cell scheme needed full quadrature against never arises

## higher order

`space_order` is $2r$ for a stencil radius $r$, and any even order is allowed. Each staggered derivative widens to $2r$ taps with the coefficients of [stencils](stencils.md) `staggered_weights`, graded down towards a wall so no tap leaves the domain, exactly as the scalar flux grades. The cost is a handful of $2r$-tap differences per node, so order 6 runs within a factor of the memory-bound floor of order 2 and the choice of order is no longer a cost question

Two prices remain, and they are the honest limits of a wide stencil rather than of this scheme:

- the leapfrog stays $O\left(\Delta t^2\right)$ in time, so at a fixed Courant number the asymptotic rate is 2 whatever the spatial order; the spatial order pays through **dispersion** in the practical points-per-wavelength regime, which is the classic $(2,4)$ trade of seismology
- a wide stencil still reaches past the harmonic mean into a void, so a design floor at $\gamma=10^{-3}$ costs stable timestep at order 4 where order 2 is untouched. The collapse is far milder than the cell gather's, but `stable_timestep(sim, indicator)` remains the arbiter: it power-iterates the step kernel itself, so it is exact for any order, material and boundary layout, where `stable_dt` knows only the speed and the spacing. A number far below it is the signal to raise the design floor or drop `space_order`, not to shrink `dt`

The reconstruction strip of [sensitivity](sensitivity.md) follows `reach` $=2r-1$ rather than $r$: the strain taps compound with the divergence taps, and a strip sized on $r$ alone would let the reverse march read a damped node

## members

| member | signature | description |
|---|---|---|
| constructor | `ElasticWave(Nx, dx, N, dt, threads, density=None, wavespeed_p=None, wavespeed_s=None, plane="strain", relaxation=None)` | on top of [Simulation](wave.md), the two speeds, the background density and the material's attenuation, rejecting a shear speed above the pressure one, any face that is not `Traction` or `Clamped` and a relaxation beside a pml |
| attenuation | `Relaxation(quality, band, reference, mechanisms=3)` | the frozen record `relaxation` takes: $Q$ over a band, and the frequency at which the speeds are phase speeds |
| relaxation fit | `zener(relaxation, dt)` | the `Zener` rates, weights and `unrelaxed` factor, fitted where the step sees the band, raising on a negative weight |
| unrelaxed moduli | `unrelaxed` | $M_U/M\left(f_r\right)$, the factor on `lame` and `shear`, 1 without a relaxation |
| staggering | `component_offsets` | component `c` half a node up axis `c`, what [distribute](utils.md) shifts by |
| stencil radius | `radius` | half `space_order`, each derivative reaching twice it |
| strip radius | `reach` | $2r-1$, the nodes one step reads past a point |
| materials | `build_materials(indicator)` | the design field alone, and `damping`: the kernels average $\gamma$ onto the stress and component points themselves, so no material grid is stored beside it |
| point inertia | `point_inverse_inertia(gamma, lin_index)` | $1/\left(\rho_0W\bar{\gamma}\right)$ on the component points `lin_index`, zero where a clamped wall holds them, the value the kernels form, for the excitation weights |
| step | `define_step(kernels, mat)` | the two-launch closure: the stress kernel, then the update, each followed by its [pml](boundary.md) terms when `pml` is set, the stress kernel relaxing the stresses under a `relaxation`; without either, where the fields spill out of L2 and the stencil is narrow enough for the tile (2D, and order 2 in 3D), one fused launch instead (`FUSED_TILE`) |
| blocked step | `define_blocked_step(mat, source, sensors)` | in 2D float32 without a pml or a relaxation, a `Blocked` closure advancing several steps a launch, the excitation and the record inside the tile, bitwise the pair launched step by step: on a grid whose tiles fill at most `BLOCKED_WAVES` waves of multiprocessors (four at order 2, one at order 4) the candidate of `BLOCKED` with the fewest waves, at order 2 past 2.25 times L2 `LARGE_BLOCKED`, which beats the fused launch there; else `None` |
| adjoint step | `define_adjoint_step(kernels, sens_kernels, mat)` | the step itself, a relaxation's memory included, or with a pml its transpose, which pushes each memory back through the other pass |
| adjoint gradient | `define_adjoint_gradient(kernels, mat, grads)` | the adjoint step carrying both gradients, compiled with `-DGRADIENT` on either tier of the step, a pml included: the stress pass adds the stiffness densities of `l1` against the forward field, under a relaxation of its relaxed strains, the update the inertia density summed by parts in time; `sensitivity` takes it since `fuses_adjoint` is set |
| pml margin | `pml_margin` | $r-1$, the undamped layer nodes before the front, since a corrected strain is read that far |
| pml arguments | `step_args(mat)`, `stress_material(mat)` | the stress scratch and the argument lists the stress and update kernels, and their pml terms, share |
| inertia | `inverse_inertia(indicator)` | nodal $1/\left(\gamma\rho_0W\right)$, what a [sponge](boundary.md) scales its damping by |
| coefficients | `parametrization_jacobian(indicator)` | $\left(1,\,1\right)$, since $\gamma$ scales inertia and stiffness alike |
| per-axis factors | `step_factors()` | $\Delta t^2/\Delta x_d$, the strain carrying the other $1/\Delta x$ |
| source scaling | `source_factor()` | $1$; the excitation weights divide by the volume the kernel inertia leaves out |
| lattice geometry | `wall_weights`, `point_average`, `pair_average` | the point families' cell weights and material means, shared with [maxwell](maxwell.md) and therefore living in [wave](wave.md) |
| timestep | `stable_timestep(sim, indicator, ...)` | the largest stable step, measured on the step kernel; also in [wave](wave.md), since it names no elastic concept |
| stiffness matrix | `voigt(ndim, lame, shear, plane="strain")` | the isotropic Voigt matrix, `plane` selecting strain or stress in 2D |

## parametrization

A single indicator scales the density while both wave speeds stay fixed, so $\mathbf{C}=\gamma\rho_0\tilde{\mathbf{C}}$ scales inertia and stiffness alike and the stable timestep does not move as the design does

$$\tilde{C}_{ijkl}=\left(c_p^2-2c_s^2\right)\delta_{ij}\delta_{kl}+c_s^2\delta_{ik}\delta_{jl}+c_s^2\delta_{il}\delta_{jk}$$

with the Lamé parameters recovered as `lame` $\lambda=\rho_0\left(c_p^2-2c_s^2\right)$ and `shear` $\mu=\rho_0c_s^2$, both times `unrelaxed` under a [relaxation](#relaxation). This is the parametrization ultrasonic full waveform inversion reaches for, following [Bürchner et al. 2025](https://doi.org/10.1016/j.ultras.2025.107705), since a void is a density contrast at unchanged speeds and the optimization stays on one field

`plane` is a host-side choice and costs the kernel nothing: plane strain takes $\lambda$ as it is, plane stress replaces it by $2\lambda\mu/\left(\lambda+2\mu\right)$, and 3D rejects anything but strain

## boundary conditions

| condition | meaning | cost |
|---|---|---|
| `Traction` | $\boldsymbol{\sigma}\cdot\mathbf{n}=0$, the default: the wall stress point condenses its axis out of the coupling | no kernel and no launch |
| `Clamped` | $\mathbf{u}=0$, the tangential wall unknowns held by a zeroed inverse inertia, the normal strain folded antisymmetrically about the wall | no kernel and no launch |
| `sponge` | the same damping field the pressure equation takes, shared by every component | as [boundary](boundary.md) |
| `Pml` | a matched layer behind `Clamped` faces, `frequency` required, a free surface may cross it, no relaxation beside it | a strain and a divergence memory per lined axis, two launches a step |

A traction wall condenses each graded-out axis by the plane stress reduction $\lambda\to2\lambda\mu/\left(\lambda+2\mu\right)$, so the surface-tangential stiffness is the statically condensed one rather than merely the interior one with a row deleted. The wall-normal component carries no unknown on the wall itself, so a source or receiver coordinate placed **on** a traction surface lands on the unknown half a cell inside, which is the staggered convention for a surface force

A sponge behind an elastic wall has to swallow the shear and surface waves as well as the pressure one, and the shear wavelength is $c_p/c_s$ shorter at the same frequency, so a layer sized on the pressure wavelength leaks. Size it on $c_s$ and expect what is left to show in the tangential component

## relaxation

`relaxation` gives the material its own attenuation: a generalized Zener body, whose memory variables hold the quality factor over a band, the standard of time-domain seismic modelling going back to [Emmerich & Korn 1987](https://doi.org/10.1190/1.1442386) and put on a staggered grid by [Robertsson, Blanch & Symes 1994](https://doi.org/10.1190/1.1443701). The damping $d$ of a [sponge](boundary.md) is no substitute: it decays every frequency at the same rate in time, so its $Q$ grows with frequency, where a material loses about the same fraction per cycle

$$\boldsymbol{\sigma}=\boldsymbol{\sigma}_U-\sum_l\boldsymbol{\zeta}_l,\qquad\dot{\boldsymbol{\zeta}}_l+\omega_l\boldsymbol{\zeta}_l=\omega_lY_l\,\boldsymbol{\sigma}_U,\qquad M\left(\omega\right)=M_U\left(1-\sum_l\frac{Y_l\,\omega_l}{\omega_l+i\omega}\right),\qquad Q=\frac{\textrm{Re}\,M}{\textrm{Im}\,M}$$

with the unrelaxed stress $\boldsymbol{\sigma}_U=\mathbf{C}:\boldsymbol{\varepsilon}$, the `quality` $Q$ held over the `band` $\left(f_\textrm{low},f_\textrm{high}\right)$, the `mechanisms` relaxation `rates` $\omega_l$ and `weights` $Y_l$ that `zener` fits, and the `reference` frequency $f_r$ at which `wavespeed_p` and `wavespeed_s` are the phase speeds. Both moduli relax alike, so $Q_P=Q_S$, and the memory variables relax the whole stress tensor, the condensation at a traction wall included

The rates are log-spaced over the band and extended by half their spacing past either end, so every frequency of the band lies between two Debye peaks; $\textrm{Re}\,M/Q=\textrm{Im}\,M$ is linear in the weights and solved in least squares, and a negative weight, which no passive material has, raises. From 2.5 mechanisms a decade $Q$ holds within about 1%, and fewer leave it rippling. Causality ties a dispersion to the loss, so the step reads the **unrelaxed** moduli, `lame` and `shear` times `unrelaxed` $=M_U/M\left(f_r\right)$, which puts the phase speeds at $f_r$ where the constructor states them

The memory variables live on the stress points, `mechanisms` grids per Voigt row, and advance in the stress pass by the trapezoidal rule centred on the step

$$\boldsymbol{\zeta}_l^{n+\frac12}=a_l\boldsymbol{\zeta}_l^{n-\frac12}+b_l\boldsymbol{\sigma}_U^n,\qquad\boldsymbol{\sigma}^n=\boldsymbol{\sigma}_U^n-\frac12\sum_l\left(\boldsymbol{\zeta}_l^{n+\frac12}+\boldsymbol{\zeta}_l^{n-\frac12}\right),\qquad a_l=\frac{1-\omega_l\Delta t/2}{1+\omega_l\Delta t/2},\quad b_l=\frac{\omega_l\Delta t\,Y_l}{1+\omega_l\Delta t/2}$$

which is exactly the continuous body at the warped frequency $\tilde{\omega}=2\tan\left(\omega\Delta t/2\right)/\Delta t$. `zener` fits there, so the discrete $Q$ is the target rather than an approximation of it. The memory is carried in the step closure's `memory`, so `march`, the exchange of [decomposition](decomposition.md) and the stored history hold it without knowing it is there

The relaxed recursion is its own transpose: rescaling each memory variable by $b_l/h_l$, with $h_l=\left(1+a_l\right)/2$, turns the transposed recursion into the forward one, as the time-reversed adjoint of the continuous anelastic equation is the forward equation ([Komatitsch et al. 2016](https://doi.org/10.1093/gji/ggw224)). `define_adjoint_step` is therefore the step itself. `define_adjoint_gradient` holds the memory in strains instead, which is equivalent since the stiffness is pointwise and constant in time, so the stiffness density pairs the relaxed adjoint strain with the stored forward one and the history stays $u$ alone

The price is memory and the fast paths: a relaxed step reads and writes `mechanisms` times `nvoigt` extra grids and takes the plain stress and update pair, since the fused and blocked steps refuse it, as does a `pml`, so the faces take a sponge. The memory cannot be run backwards, its decay turning into growth at the rates $\omega_l$, so of the [sensitivity](sensitivity.md) variants only `sensitivity` and `source_sensitivity` take it. The step also reads the unrelaxed speed, a few percent above `wavespeed_p` at $Q$ of a few tens, which the safety factor on the timestep has to cover

## limits

In 1D on a uniform material the staggered scheme is the [scalar](scalar.md) stencil on the half-shifted grid, unknown for unknown, which is the one configuration where the whole vector pipeline is checked against an already-trusted one. On a varying material the two sample $\gamma$ on offset grids and agree only to discretization order

The staggering is also what this scheme cannot escape: a general anisotropic $\mathbf{C}$ couples strains that live on different points, so anisotropy belongs to the collocated [AnisotropicElasticWave](anisotropic.md), which trades the linear-in-radius cost for it

The kernels are documented in [forward CUDA](cuda_elastic.md) and [backward CUDA](cuda_elastic_sensitivity.md)
