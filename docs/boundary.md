# Boundary

**Boundary** conditions close the domain, one per face — a face being an outer boundary of the grid, encoded as $2d+\textrm{side}$ with the axis $d$ and side 0 low or 1 high

A condition is a declarative marker naming a kernel and nothing more, so a setup can place one per face in `Simulation.boundary` long before anything is compiled. `define_boundary` turns the markers into launch closures once `compile_kernels` has run, which is why `simulate` needs no separate preparation phase

| member | signature | description |
|---|---|---|
| condition | `BoundaryCondition(kernel)` | frozen marker holding the name of the kernel that enforces it, so it can key the grouping below |
| launch | `BoundaryCondition.define(sim, kernels, faces)` | returns `step(u)`, applying this condition to every face of `faces` in one launch |
| reflecting | `Neumann` | zero flux, $\partial u/\partial n=0$, the default on every face |
| absorbing pressure | `Dirichlet` | zero field, $u=0$, the pressure-release or free-surface wall |
| random absorber | `random_layer(sim, width, contrast=0.5, faces=None, correlation=1, rng=None)` | a multiplicative field over the padded grid, 1 outside the `width` nodes behind the named faces and randomly perturbed inside them, for a driver to multiply into its indicator |
| normalization | `canonical_boundary(boundary, ndim)` | expands the shorthands — `None`, one condition, one per axis — into the `((low, high),) * ndim` form `Simulation` stores |
| dispatch | `define_boundary(sim, kernels)` | groups the faces by condition and returns `bc_step(u)`, which runs one launch per **distinct** condition |

Both walls are the discrete method of images and act on the ghost ring only, mirroring the second-interior node onto it — evenly for `Neumann`, oddly for `Dirichlet` — so the interior kernel needs no boundary branch at all. [forward CUDA](cuda_wave.md) carries the two kernels and the thread-to-ghost mapping

Grouping by condition rather than by face is what keeps the common case cheap: a uniform boundary is one launch however many faces it covers, and a mixed one costs one launch per distinct condition rather than $2\,\textrm{ndim}$. The faces travel as a bitmask, so no device array has to be allocated or kept alive for the dispatch

Only these two *conditions* are implemented. A true absorbing or radiating condition would need state on the boundary layer rather than a ghost mirror, so it does not fit the marker-plus-kernel shape and is left out rather than approximated — `random_layer` opens the domain in the material instead, which is why it is a field builder and not a `BoundaryCondition`

## random layer

**A random layer** absorbs by scattering rather than by dissipating: the `width` nodes behind a reflecting face carry a randomized material, so what comes back out of them is an incoherent coda instead of a specular echo, following [Shen & Clapp 2015](https://doi.org/10.1190/geo2014-0542.1)

$$\gamma\left(\mathbf{x}\right)=\gamma_0\left(\mathbf{x}\right)\left(1+\varepsilon\,w\left(\mathbf{x}\right)\xi\left(\mathbf{x}\right)\right)$$

with the design field $\gamma_0$ the driver already had, the peak perturbation `contrast` $\varepsilon$, the taper $w$ rising linearly from 0 at the inner edge of the layer to 1 on the wall node, and $\xi$ drawn uniformly from $[-1,1]$ on blocks of `correlation` nodes from the generator `rng`. Overlapping layers take the larger taper, so a corner is perturbed once rather than once per face

The layer multiplies the indicator, so **the formulation decides what gets randomized** and one call serves both

| formulation | $\gamma$ scales | the layer randomizes |
|---|---|---|
| `ScalarWave` | the density $\rho=\gamma\rho_0$, and with it the impedance | the impedance, at $c\equiv c_0$ everywhere |
| `AcousticWave` | $1/\rho$ and $1/\kappa$ alike | the wave speed $c=\sqrt{\kappa/\rho}$ |

Setting a wave-speed layer up on `AcousticWave` means holding the two densities equal and putting the two bulk moduli either side of the background, the same `contrast` $\varepsilon$ serving as the material contrast

$$\rho_1=\rho_2=\rho,\qquad\kappa_{1,2}=\frac{\rho c_0^2}{1\pm\varepsilon},\qquad\gamma_0=\frac{1}{2}$$

so the interior runs at $c_0$ and the layer's $\pm\varepsilon$ swing in $\gamma$ moves the slowness $1/c^2$ by $\mp\varepsilon^2$ about $1/c_0^2$. The wave speed then ranges over $c_0/\sqrt{1+\varepsilon^2}$ to $c_0/\sqrt{1-\varepsilon^2}$, and it is that upper end `stable_dt` has to be given — a symmetric perturbation of the slowness is a lopsided one of the wave speed, so the layer costs timestep as well as cells

Prefer the wave speed where the setup allows it. Randomizing the impedance alone still backscatters, but every scatterer sits at the traveltime the background gives it, so the return keeps more coherence than a layer that distorts the traveltimes as well

The taper is what makes the layer usable at all: a perturbation that switched on at full strength would put a coherent specular reflector at the inner edge, which is the echo the layer exists to remove. Everything else is thickness — the layer has to be many wavelengths deep before the backscatter accumulates, and a dozen is a reasonable starting point. `correlation` tunes the scatterer size and is a weak lever for a narrowband source, where a value near the half wavelength is as good as any; the multi-scale zones [Shen & Clapp 2015](https://doi.org/10.1190/geo2014-0542.1) build are aimed at the broadband low-frequency wavefield of an inversion, where one scale cannot serve every wavelength

This is not a low-reflection boundary, and reaching for it as one will disappoint: it trades a coherent echo for an incoherent coda of the same order, and the price of even that is grid — the layer is added *outside* the region of interest, so a dozen wavelengths per face is a domain several times the cells. What it buys instead is losslessness. A damping sponge absorbs far better and is rejected outright by both [sensitivity](sensitivity.md) variants, whereas a random layer leaves the operator exactly time-reversible, so the memory-efficient gradient still applies and the leftover coda averages out of a gradient stacked over shots rather than biasing it

`examples/forward/acoustic_2D_absorbing_bcs.py` is the demonstration and `examples/forward/scalar_2D_absorbing_bcs.py` its impedance-only counterpart: a layer on both faces of one axis and a reflecting wall on the other, so one snapshot carries the specular echo the reflecting axis returns beside the coda the lined axis returns in its place

An inversion has to freeze the layer: it is part of the indicator, so `misfit_gradient` returns a gradient over those nodes too, and the driver masks them rather than letting the optimizer redesign its own boundary
