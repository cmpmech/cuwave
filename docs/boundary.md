# Boundary

**Boundary** conditions close the domain, one per face — a face being an outer boundary of the grid, encoded as $2d+\textrm{side}$ with the axis $d$ and side 0 low or 1 high

A condition is a declarative marker naming a kernel and nothing more, so a setup can place one per face in `Simulation.boundary` long before anything is compiled. `define_boundary` turns the markers into launch closures once `compile_kernels` has run, which is why `simulate` needs no separate preparation phase

| member | signature | description |
|---|---|---|
| condition | `BoundaryCondition(kernel)` | frozen marker holding the name of the kernel that enforces it, so it can key the grouping below |
| launch | `BoundaryCondition.define(sim, kernels, faces)` | returns `step(u)`, applying this condition to every face of `faces` in one launch |
| reflecting | `Neumann` | zero flux, $\partial u/\partial n=0$, the default on every face |
| absorbing pressure | `Dirichlet` | zero field, $u=0$, the pressure-release or free-surface wall |
| random absorber | `random_layer(sim, indicator, width, correlation, bounds=(0.0, 1.0), faces=None, rng=None)` | a copy of `indicator` with the `width` nodes behind the named faces redrawn at random, for a driver to hand `simulate` in its place |
| normalization | `canonical_boundary(boundary, ndim)` | expands the shorthands — `None`, one condition, one per axis — into the `((low, high),) * ndim` form `Simulation` stores |
| dispatch | `define_boundary(sim, kernels)` | groups the faces by condition and returns `bc_step(u)`, which runs one launch per **distinct** condition |

Both walls are the discrete method of images and act on the ghost ring only, mirroring the second-interior node onto it — evenly for `Neumann`, oddly for `Dirichlet` — so the interior kernel needs no boundary branch at all. [forward CUDA](cuda_wave.md) carries the two kernels and the thread-to-ghost mapping

Grouping by condition rather than by face is what keeps the common case cheap: a uniform boundary is one launch however many faces it covers, and a mixed one costs one launch per distinct condition rather than $2\,\textrm{ndim}$. The faces travel as a bitmask, so no device array has to be allocated or kept alive for the dispatch

Only these two *conditions* are implemented. A true absorbing or radiating condition would need state on the boundary layer rather than a ghost mirror, so it does not fit the marker-plus-kernel shape and is left out rather than approximated — `random_layer` opens the domain in the material instead, which is why it is a field builder and not a `BoundaryCondition`

## random layer

**A random layer** absorbs by scattering rather than by dissipating: the `width` nodes behind a reflecting face carry a randomized material, so what comes back out of them is an incoherent coda instead of a specular echo, following [Shen & Clapp 2015](https://doi.org/10.1190/geo2014-0542.1)

$$\gamma\left(\mathbf{x}\right)\sim\mathcal{U}\left(\gamma_\textrm{lo},\gamma_\textrm{hi}\right)\qquad\textrm{one draw per grain of }\ell^{\,\textrm{ndim}}\textrm{ nodes}$$

with the `bounds` $\left(\gamma_\textrm{lo},\gamma_\textrm{hi}\right)$ the draw spans, the grain size `correlation` $\ell$ sharing one value across a block of that many nodes per axis, and the generator `rng` a driver fixes so its layer reproduces. Outside the layer the field is the `indicator` handed in, untouched, so the layer is written into a design field rather than multiplied onto it

The layer replaces the indicator, so **the formulation decides what gets randomized** and one call serves both

| formulation | $\gamma$ scales | the layer randomizes |
|---|---|---|
| `ScalarWave` | the density $\rho=\gamma\rho_0$, and with it the impedance | the impedance, at $c\equiv c_0$ everywhere |
| `AcousticWave` | $1/\rho$ and $1/\kappa$ alike | the wave speed $c=\sqrt{\kappa/\rho}$ |

Prefer the wave speed. Randomizing the impedance alone still backscatters, but every scatterer sits at the traveltime the background gives it, so the return keeps more coherence than a layer that distorts the traveltimes as well

`AcousticWave` reaches it with equal densities and the two phases placed at the wave speeds the layer is to span, so the draw covers the whole indicator range and the interior is one point inside it

$$\rho_1=\rho_2=\rho,\qquad\kappa_1=\rho c_\textrm{min}^2,\qquad\kappa_2=\rho c_\textrm{max}^2,\qquad\gamma_0=\frac{c_\textrm{min}^{-2}-c_0^{-2}}{c_\textrm{min}^{-2}-c_\textrm{max}^{-2}}$$

with `bounds` then $\left(0,1\right)$, the interior at $\gamma_0$, and $c_\textrm{max}$ the speed `stable_dt` has to be given

### the knobs

`correlation` decides whether the layer works at all, because a wave averages the material over its own wavelength: grains much finer than that average into a constant and let the wavefront through coherently, however large the contrast. **Take the grain near half the dominant wavelength.** The single-node draw that serves reverse time migration is far too fine for the long wavelengths of an inversion, and matching $\ell$ to the wavelength instead is the point of [Shen & Clapp 2015](https://doi.org/10.1190/geo2014-0542.1). One grain size serves a narrowband source; a broadband one has low-frequency components that outrun any single scale

`bounds` should be as wide as the timestep tolerates, and **wide asymmetrically**: the slow half of the range does the scattering, since a slower layer holds more wavelengths across the same nodes, while only the fast end enters the CFL condition. A range symmetric about the background wastes both — it pays for a fast tail and forfeits the slow one

`width` matters least. One dominant wavelength per face already gets most of the effect and two is comfortable; beyond that the layer delays the coda rather than weakening it. There is no taper, and none is wanted: the draw is uniform, so a long wavelength sees the layer's mean and its front is not a coherent reflector to begin with, while ramping the contrast in would only dilute the scattering over the nodes it covers

### what it buys

**A random layer cannot reduce the reflected energy**, and reaching for it as a low-reflection boundary will disappoint. It is lossless by construction — which is the reason it exists — so everything that enters the layer comes back out, and all that changes is that it comes back incoherent. Judge it by stacking realizations rather than by amplitude: a specular echo survives that average untouched, while a scattered one falls towards $1/\sqrt{M}$ over $M$ draws. That incoherence is what buys the gradient. A damping sponge absorbs far better and is rejected outright by both [sensitivity](sensitivity.md) variants, whereas a random layer leaves the operator exactly time-reversible and its coda decorrelates from the adjoint field instead of biasing the gradient

Where it earns its keep is a **thin** pad. At one to two wavelengths per face a random layer leaves a cleaner gradient than a reflecting wall the same distance out, and redrawing it every iteration — the same stack, spread over the optimization instead of over one figure — takes most of what remains. Past about four wavelengths the ranking inverts: a plain reflecting pad that far out returns its echo after the residual has died away, so randomizing it only injects noise. Reach for the layer where memory is the binding constraint, which is the case it was built for, and for plain padding where grid is cheap

`examples/forward/acoustic_2D_absorbing_bcs.py` is the demonstration and `examples/forward/scalar_2D_absorbing_bcs.py` its impedance-only counterpart: a layer on both faces of one axis and a reflecting wall on the other, so one snapshot carries the specular echo the reflecting axis returns beside the coda the lined axis returns in its place

An inversion has to freeze the layer: it is part of the indicator, so `misfit_gradient` returns a gradient over those nodes too, and the driver masks them rather than letting the optimizer redesign its own boundary. Redrawing it per iteration is the cheaper half of the same idea, since a fresh realization each time is the stack that averages the coda out
