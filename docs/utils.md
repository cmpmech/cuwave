# Utils

**Utils** carries the pieces a driver needs around the solver: putting a source at a physical coordinate, reading receivers back off the grid, and running the misfit and its gradient over a list of shots

Sources and receivers are placed by **coordinate**, not by node index, and multilinearly interpolated onto the surrounding $2^\textrm{ndim}$ corners. So a transducer array sits at the same physical points on every resolution, which is what makes two grids comparable at all. A staggered unknown declares `component_offsets` on its [Simulation](wave.md), and each component is then interpolated on its own shifted grid; a coordinate on a wall along a shifted axis lands on the unknown half a cell inside, the staggered convention for a surface force

| member | signature | description |
|---|---|---|
| interior | `interior_slice(sim)` | the index tuple selecting the interior nodes, dropping the ghost ring and the padding tail in one slice |
| segmentation | `threshold(field, eta=0.5, low=0.0, high=1.0, dtype=None)` | snaps a grey design to its two materials about `eta`, keeping the dtype of `field` |
| resampling | `resample(signal, dt, dt_new, N_new=None)` | linear resampling along the leading axis, on either array module, zero past the end of the original span |
| coordinates | `line(start, stop, count)` | `count` coordinates evenly spaced from `start` to `stop`, endpoints included |
| interpolation | `distribute(sim, coords, direction=None)` | the `(nodes, weights)` of the surrounding cell corners, per component where `component_offsets` shifts them, raising for a coordinate outside the domain |
| source | `point_source(sim, coords, signal, direction=None)` | a `Source` firing `signal` at `coords`, divided by the cell volume so the amplitude does not depend on the spacing |
| source chain rule | `collect_source(sim, coords, columns, direction=None)` | the transpose of `point_source`: an $(N,\,\textrm{num}\cdot2^\textrm{ndim})$ node gradient contracted onto the signal of each coordinate |
| shot list | `shots(sim, coords, signal, direction=None)` | one single-coordinate `Source` per coordinate, which is the shot list of an inversion |
| stacking | `stack(sources)` | concatenates positions and signal columns, firing several shots in a single simulation |
| receivers | `Sensors(sim, coords, direction=None)` | precomputes the interpolation once; `traces(record)` reduces a node record onto receivers and `scatter(dphi)` is its transpose |
| lifting | `Sensors.objective(objective)` | wraps a receiver-space objective into the node space [sensitivity](sensitivity.md) wants |
| experiment | `measure(sim, sources, indicator, sensors)` | the receiver traces per shot, i.e. the synthetic experiment an inversion is fitted to |
| cost | `misfit(sim, sources, indicator, sensors, observed, objective=l2_misfit)` | the summed cost alone, forward passes only |
| cost and gradient | `misfit_gradient(sim, sources, indicator, sensors, observed, objective=l2_misfit, adjoint=sensitivity)` | the same cost with its derivative, already reparametrized from $(m,k)$ onto the indicator |
| objective | `energy(sim)` | factory for $J=\tfrac{1}{2}\Delta t\prod_d\Delta x_d\sum p^2$, the acoustic energy reaching the sensor nodes |
| objective | `intensity(sim, frequencies, weights=None)` | factory for $J=\sum_{fc}w_{fc}\lvert\hat{u}_{fc}\rvert^2$, the spectral intensity at the sensor nodes, `weights` pairing each frequency with the port it is scored on |
| cost | `response(sim, source, indicator, sensors, objective)` | one forward pass, returning the cost together with the field at the last step |
| cost and gradient | `response_gradient(sim, source, indicator, sensors, objective, adjoint=sensitivity)` | the same cost with its derivative, reparametrized onto the indicator as above |

`direction` is what makes the same helpers serve a vector unknown: it is optional for the pressure classes, whose single component needs none, and required for [elastic](elastic.md) and [maxwell](maxwell.md), where a source or a receiver has to say which component it drives or reads. `distribute` carries the same argument underneath all four, so the component offsets of a staggered layout are honoured once rather than per caller

The three applications get the same pair twice: `misfit` / `misfit_gradient` sums over a shot list against measured data for [fwi](fwi.md), `response` / `response_gradient` scores one shot against a design objective for [tato](tato.md) and [tpto](tpto.md). Both route the chain rule from $(m,k)$ onto the indicator through the same private `_reparametrize`, so a new `Simulation` subclass changes neither

`response` returns the field alongside the cost because the thresholded design is re-simulated precisely to be looked at: the number and the picture come from the same solve, and cannot drift apart

Both gradients take the adjoint variant itself as an argument, so a sponged run swaps in `reconstruction_sensitivity` without touching the objective, the shot loop or the chain rule; see [sensitivity](sensitivity.md) on which variant fits

`misfit` and `misfit_gradient` are the pair a line search needs: a trial step evaluates the first, which is one forward pass per shot, and only an accepted iteration pays for the second, which adds an adjoint pass; see the [optimization](optimization.md) line search on what that ratio buys

`stack` trades resolution for cost. A stacked shot is one simulation and one record instead of `len(sources)`, but the gradient it produces is the superposition of the individual ones and cannot be separated again; encode the shots by scaling their signals (random signs, phase shifts) before stacking if the crosstalk matters

Receiver interpolation is the transpose of source distribution, deliberately: `scatter` is exactly `traces` run backwards, so an objective defined on receivers differentiates correctly without the driver forming anything on the grid

`intensity` is the spectral sibling of `energy`, and it is what a time-domain solver needs in order to answer a frequency-domain question. An objective already receives the whole $(N,\,\textrm{num sensors})$ record rather than one step of it, so a discrete Fourier transform is an ordinary objective and needs nothing from the adjoint: the transform is linear, so its derivative is the same pair of tables read backwards. The tables are built in double and stored in the simulation's own precision, since the phase reaches $10^5$ radians over a long run

Several frequencies cost one run, not one run each, which is the whole reason [tpto](tpto.md) is transient: `weights` is $(\textrm{num frequencies},\,\textrm{num sensors})$, so a demultiplexer scoring two wavelengths on two ports is one broadband simulation and a two-row matrix. The run has to be long enough for the transform to resolve them, the bin spacing being $1/\left(N\Delta t\right)$
