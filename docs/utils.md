# Utils

**Utils** carries the pieces a driver needs around the solver: putting a source at a physical coordinate, reading receivers back off the grid, and running the misfit and its gradient over a list of shots

Sources and receivers are placed by **coordinate**, not by node index, and multilinearly interpolated onto the surrounding $2^\textrm{ndim}$ corners. So a transducer array sits at the same physical points on every resolution, which is what makes two grids comparable at all

| member | signature | description |
|---|---|---|
| interior | `interior_slice(sim)` | the index tuple selecting the interior nodes, dropping the ghost ring and the padding tail in one slice |
| segmentation | `threshold(field, eta=0.5, low=0.0, high=1.0, dtype=None)` | snaps a grey design to its two materials about `eta`, keeping the dtype of `field` |
| resampling | `resample(signal, dt, dt_new, N_new=None)` | linear resampling along the leading axis, on either array module, zero past the end of the original span |
| coordinates | `line(start, stop, count)` | `count` coordinates evenly spaced from `start` to `stop`, endpoints included |
| interpolation | `distribute(sim, coords)` | the `(nodes, weights)` of the surrounding cell corners, raising for a coordinate outside the domain |
| source | `point_source(sim, coords, signal)` | a `Source` firing `signal` at `coords`, divided by the cell volume so the amplitude does not depend on the spacing |
| shot list | `shots(sim, coords, signal)` | one single-coordinate `Source` per coordinate, which is the shot list of an inversion |
| stacking | `stack(sources)` | concatenates positions and signal columns, firing several shots in a single simulation |
| receivers | `Sensors(sim, coords)` | precomputes the interpolation once; `traces(record)` reduces a node record onto receivers and `scatter(dphi)` is its transpose |
| lifting | `Sensors.objective(objective)` | wraps a receiver-space objective into the node space [sensitivity](sensitivity.md) wants |
| experiment | `measure(sim, sources, indicator, sensors)` | the receiver traces per shot, i.e. the synthetic experiment an inversion is fitted to |
| cost | `misfit(sim, sources, indicator, sensors, observed, objective=l2_misfit)` | the summed cost alone, forward passes only |
| cost and gradient | `misfit_gradient(sim, sources, indicator, sensors, observed, objective=l2_misfit, adjoint=sensitivity)` | the same cost with its derivative, already reparametrized from $(m,k)$ onto the indicator |
| objective | `energy(sim)` | factory for $J=\tfrac{1}{2}\Delta t\prod_d\Delta x_d\sum p^2$, the acoustic energy reaching the sensor nodes |
| cost | `response(sim, source, indicator, sensors, objective)` | one forward pass, returning the cost together with the field at the last step |
| cost and gradient | `response_gradient(sim, source, indicator, sensors, objective, adjoint=sensitivity)` | the same cost with its derivative, reparametrized onto the indicator as above |

The two applications get the same pair twice: `misfit` / `misfit_gradient` sums over a shot list against measured data for [fwi](fwi.md), `response` / `response_gradient` scores one shot against a design objective for [tato](tato.md). Both route the chain rule from $(m,k)$ onto the indicator through the same private `_reparametrize`, so a new `Simulation` subclass changes neither

`response` returns the field alongside the cost because the thresholded design is re-simulated precisely to be looked at: the number and the picture come from the same solve, and cannot drift apart

Both gradients take the adjoint variant itself as an argument, so a sponged run swaps in `reconstruction_sensitivity` without touching the objective, the shot loop or the chain rule; see [sensitivity](sensitivity.md) on which variant fits

`misfit` and `misfit_gradient` are the pair a line search needs: a trial step evaluates the first, which is one forward pass per shot, and only an accepted iteration pays for the second, which adds an adjoint pass; see the [optimization](optimization.md) line search on what that ratio buys

`stack` trades resolution for cost. A stacked shot is one simulation and one record instead of `len(sources)`, but the gradient it produces is the superposition of the individual ones and cannot be separated again; encode the shots by scaling their signals (random signs, phase shifts) before stacking if the crosstalk matters

Receiver interpolation is the transpose of source distribution, deliberately: `scatter` is exactly `traces` run backwards, so an objective defined on receivers differentiates correctly without the driver forming anything on the grid
