# Transient Acoustic Topology Optimization

**Transient acoustic topology optimization (tato)** ([Dühring, Jensen & Sigmund 2008](https://www.sciencedirect.com/science/article/pii/S0022460X08002812)) distributes a fixed amount of material inside a design region so that a transient acoustic response is minimized: a scatterer or barrier shaped by the wave it has to stop, rather than by a rule of thumb

An optimization is set up as follows:
1. define the forward problem with [scalar](scalar.md) (here `AcousticWave`, whose indicator $\gamma$ interpolates between air and solid)
2. mark the design region and the objective region with `box`, the second turned into sensors by `nodes` ([geometry](geometry.md))
3. map the design variables through a `DensityFilter` and a `Projection` ([regularization](regularization.md))
4. iterate `response_gradient` and one `Adam` step ([utils](utils.md)), sharpening the projection on a continuation schedule
5. snap the final design to 0/1 with `threshold` and re-simulate it with `response` to report what it achieves

$$C=\frac{1}{2}\int_{\Omega_\textrm{target}}\int_0^Tp^2\,\textrm{d}t\,\textrm{d}\Omega$$

with the pressure $p$ over the target region $\Omega_\textrm{target}$ (the acoustic energy leaking into the box), evaluated on the sensor nodes that tile it and scaled by $\Delta t\prod_d\Delta x_d$ so the sum is an integral. Maximizing instead is a sign change on the gradient, which `examples/tato/acoustic_optimization_2D.py` exposes as `MINIMIZE`

The design variables are filtered then projected then masked to the region, so the chain rule runs backwards through the same three maps before it reaches the optimizer: the driver is the only place that ordering exists, since each `Regularization` knows only its own adjoint

The projection is sharpened over the run rather than held fixed: at high $\beta$ from the start the derivative vanishes away from the threshold and the design barely moves, at low $\beta$ throughout it converges grey. The price is an objective that changes between iterations, so the history need not decrease across a step in $\beta$

## thresholded evaluation

**The design the optimizer converges to is grey and no structure achieves its objective.** Every iteration sees a projected field somewhere in $[0,1]$, and the material interpolation gives an intermediate $\gamma$ an intermediate density and bulk modulus that no mixture of air and solid has. Only the thresholded design can be built, so it is the one a run reports:
$$\gamma_\textrm{final}=\begin{cases}0&\gamma<\eta\\1&\gamma\ge\eta\end{cases}\qquad\textrm{price}=10\log_{10}\frac{C(\gamma_\textrm{final})}{C(\gamma)}$$
with the projected design $\gamma$, the cut $\eta$ at the midpoint of the projection's own range, and the two costs from a `response` solve each. The driver prints that **discretization price** next to `non_discreteness` ([evals](evals.md)), since the two move together: a design already near 0/1 has nothing to lose by being snapped, and a large price is a report that the continuation schedule ended too soft rather than that the optimization failed

The price is paid once, at the end, and is not optimized against: thresholding is not differentiable, so folding it into the loop would leave the gradient with nothing to say. Sharpening $\beta$ over the run is the differentiable approximation of it, which is what makes the schedule a convergence requirement and not a cosmetic one
