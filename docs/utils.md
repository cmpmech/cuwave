# Utils

**Utils** carries the pieces a driver needs around the solver: putting a source at a physical coordinate, reading receivers back off the grid, and running the misfit and its gradient over a list of shots

Sources and receivers are placed by **coordinate**, not by node index, and multilinearly interpolated onto the surrounding $2^\textrm{ndim}$ corners. So a transducer array sits at the same physical points on every resolution, which is what makes two grids comparable at all

| member | signature | description |
|---|---|---|
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
| cost and gradient | `misfit_gradient(sim, sources, indicator, sensors, observed, objective=l2_misfit)` | the same cost with its derivative, already reparametrized from $(m,k)$ onto the indicator |

`misfit` and `misfit_gradient` are the pair a line search needs: a trial step evaluates the first, which is one forward pass per shot, and only an accepted iteration pays for the second, which adds an adjoint pass; see the [optimization](optimization.md) line search on what that ratio buys

`stack` trades resolution for cost. A stacked shot is one simulation and one record instead of `len(sources)`, but the gradient it produces is the superposition of the individual ones and cannot be separated again; encode the shots by scaling their signals (random signs, phase shifts) before stacking if the crosstalk matters

Receiver interpolation is the transpose of source distribution, deliberately: `scatter` is exactly `traces` run backwards, so an objective defined on receivers differentiates correctly without the driver forming anything on the grid
