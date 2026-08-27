# Transient Acoustic Topology Optimization

**Transient acoustic topology optimization (tato)** ([Dühring, Jensen & Sigmund 2008](https://www.sciencedirect.com/science/article/pii/S0022460X08002812)) distributes a fixed amount of material inside a design region so that a transient acoustic response is minimized — a scatterer or barrier shaped by the wave it has to stop, rather than by a rule of thumb

An optimization is set up as follows:
1. define the forward problem with [wave](wave.md) (here `AcousticWave`, whose indicator $\gamma$ interpolates between air and solid)
2. mark the design region and the objective region on the grid ([geometry](geometry.md), or a coordinate box)
3. map the design variables through a `DensityFilter` and a `Projection` ([regularization](regularization.md))
4. iterate `sensitivity` and one `Adam` step, sharpening the projection on a continuation schedule
5. re-simulate the final design to report the response it achieves

$$C=\frac{1}{2}\int_{\Omega_\textrm{target}}\int_0^Tp^2\,\textrm{d}t\,\textrm{d}\Omega$$

with the pressure $p$ over the target region $\Omega_\textrm{target}$ (the acoustic energy leaking into the box), evaluated on the sensor nodes that tile it and scaled by $\Delta t\prod_d\Delta x_d$ so the sum is an integral. Maximizing instead is a sign change on the gradient, which `examples/tato/acoustic_optimization_2D.py` exposes as `MINIMIZE`

The design variables are filtered then projected then masked to the region, so the chain rule runs backwards through the same three maps before it reaches the optimizer — the driver is the only place that ordering exists, since each `Regularization` knows only its own adjoint

The projection is sharpened over the run rather than held fixed: at high $\beta$ from the start the derivative vanishes away from the threshold and the design barely moves, at low $\beta$ throughout it converges grey. The price is an objective that changes between iterations, so the history need not decrease across a step in $\beta$
