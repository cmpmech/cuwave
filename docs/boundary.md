# Boundary

**Boundary** conditions close the domain, one per face, a face being an outer boundary of the grid, encoded as $2d+\textrm{side}$ with the axis $d$ and side 0 low or 1 high

A condition is a declarative marker naming a kernel and nothing more, so a setup can place one per face in `Simulation.boundary` long before anything is compiled. `define_boundary` turns the markers into launch closures once `compile_kernels` has run, which is why `simulate` needs no separate preparation phase

| member | signature | description |
|---|---|---|
| condition | `BoundaryCondition(kernel)` | frozen marker holding the name of the kernel that enforces it, so it can key the grouping below |
| launch | `BoundaryCondition.define(sim, kernels, faces)` | returns `step(u)`, applying this condition to every face of `faces` in one launch |
| reflecting | `Neumann` | zero flux, $\partial u/\partial n=0$, the default on every face |
| absorbing pressure | `Dirichlet` | zero field, $u=0$, the pressure-release or free-surface wall |
| layout | `pad_for_sponge(Nx, dx, thickness, faces=None)` | grows a region of interest by a layer of physical `thickness` behind the named faces, returning the extent to build the simulation on, the layer width in nodes, and where the region of interest ends up |
| dissipating absorber | `sponge(sim, indicator, width, beta, faces=None)` | a damping field ramping to `beta` over the `width` nodes behind the named faces and zero elsewhere, for a driver to set as `Simulation.damping` |
| normalization | `canonical_boundary(boundary, ndim)` | expands the shorthands (`None`, one condition, one per axis) into the `((low, high),) * ndim` form `Simulation` stores |
| dispatch | `define_boundary(sim, kernels)` | groups the faces by condition and returns `bc_step(u)`, which runs one launch per **distinct** condition |

Both walls are the discrete method of images and act on the ghost ring only, mirroring the second-interior node onto it, evenly for `Neumann` and oddly for `Dirichlet`, so the interior kernel needs no boundary branch at all. [forward CUDA](cuda_wave.md) carries the two kernels and the thread-to-ghost mapping

Grouping by condition rather than by face is what keeps the common case cheap: a uniform boundary is one launch however many faces it covers, and a mixed one costs one launch per distinct condition rather than $2\,\textrm{ndim}$. The faces travel as a bitmask, so no device array has to be allocated or kept alive for the dispatch

Only these two *conditions* are implemented. A true absorbing or radiating condition would need state on the boundary layer rather than a ghost mirror, so it does not fit the marker-plus-kernel shape and is left out rather than approximated: `sponge` opens the domain in the material instead, dissipating the outgoing wave behind a face that stays reflecting, which is why it is a field builder and not a `BoundaryCondition`

## sponge

**A sponge** absorbs by dissipating: the `width` nodes behind a reflecting face carry a damping field, so the outgoing wave loses its energy there rather than returning at all

$$\beta\left(\mathbf{x}\right)=\beta_\textrm{wall}\,w\left(\mathbf{x}\right)^2,\qquad d=\frac{2m\beta}{\Delta t}$$

with the peak decay per step `beta` $\beta_\textrm{wall}$ reached at the wall node, the taper $w$ rising linearly from 0 at the inner edge of the layer, and the inertia $m$ the parametrization gives. The field is scaled by it, so one `beta` means the same decay whichever formulation is in use. [forward CUDA](cuda_wave.md) carries the $\beta$ the kernel actually reads

`beta` has an **optimum** rather than a monotone benefit: too small and the wave crosses the layer and returns off the wall behind it, too large and the damping gradient reflects it on the way in. The optimum sits at $\beta_\textrm{wall}\approx0.05$ and **stays there as the layer thickens**, so thicken the layer and leave `beta` alone. A one-wavelength layer is the exception and wants about twice that, since the same energy has half as many nodes to go into. Thickness is the currency, buying roughly a factor of five per doubling, and `beta` only spends it well or badly

What a sponge leaves is a small reflection per pass rather than a clean cut, so what becomes of that leakage decides whether it matters. **A face left reflecting traps it**: the leaked wave returns to graze the sponge again and again, and the residual grows with the length of the record instead of settling. Lining every face is worth far more than tuning the one face that is lined, so open them all where the setup allows it, and where it does not, read a long record with the accumulation in mind

The price is losslessness: a damped operator cannot be run backwards, so of the [sensitivity](sensitivity.md) variants `superposition_sensitivity` refuses one outright, while `sensitivity` takes a sponge and `reconstruction_sensitivity` records the layer it cannot reverse and replays it. Setting `Simulation.damping` therefore also picks the gradient, which is why the field is a constructor argument rather than something a call site passes. Where the operator has to stay reversible throughout there is no layer to reach for, only grid: pad the domain far enough that the wall echo arrives after the record ends

The layer is grid the simulation carries and the application does not own, which is arithmetic every driver would otherwise repeat. `pad_for_sponge` does it once: it converts a physical thickness into nodes off the finest axis, grows only the axes whose faces are named, and hands back the width `sponge` wants along with the part that is easy to get wrong: the physical origin the region of interest has moved to, and the index tuple that selects it back out of the grown grid for a design mask or a figure. A transducer coordinate that misses that shift lands inside the boundary layer

`examples/forward/scalar_2D_sponge.py` is the demonstration: it lines the two faces of axis 0 and leaves axis 1 reflecting, so one figure carries the absorbed faces beside the reflecting ones. `examples/fwi/fwi_2D_adam_sponge.py` is the same boundary under an inversion, where the layer also has to be frozen out of the design
