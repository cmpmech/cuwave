# Full Waveform Inversion

**Full waveform inversion (fwi)** recovers a material field from recorded waveforms by minimizing the misfit between measured and simulated traces, using the full recorded signal rather than picked arrival times

An inversion is set up as follows:
1. define the forward problem with [wave](wave.md) (here `ScalarWave`, whose indicator $\gamma$ scales the density)
2. place transducers with `line` and turn them into a shot list with `shots` and a receiver array with `Sensors` ([utils](utils.md))
3. produce or load the measurement, `measure` against the true field for a synthetic study
4. pick an [optimization](optimization.md) scheme and, if the problem needs it, a [regularization](regularization.md)
5. iterate `misfit_gradient` and one optimizer step, clipping the design back into its bounds
6. score the result against the truth with [evals](evals.md), both raw and snapped to two materials by `threshold` on the interior nodes `interior_slice` selects

$$C=\frac{1}{2}\sum_\textrm{shots}\sum_t\sum_r\left(u_r^t-u_{r,\textrm{obs}}^t\right)^2$$

with the simulated traces $u_r^t$ and the measured `observed` $u_{r,\textrm{obs}}^t$, which is `l2_misfit` and the default `objective` of `misfit_gradient`

| driver | what it adds |
|---|---|
| `examples/fwi/fwi_2D_adam.py` | the reference inversion: `Adam`, no regularization, a synthetic measurement simulated at a higher order than it is inverted at |
| `examples/fwi/fwi_2D_lbfgs.py` | `Lbfgs` with the Armijo line search in place of `Adam` |
| `examples/fwi/regularization/fwi_2D_adam_penalty.py` | a `TotalVariation` or `Tikhonov` term added to the objective |
| `examples/fwi/regularization/fwi_2D_adam_projection.py` | `DensityFilter` and `Projection` on the design, with a continuation schedule |
| `examples/fwi/regularization/fwi_2D_adam_nn.py` | the design reparametrized by a [generator](nn.md) |

An inversion is scored twice, on the recovered field and on the same field thresholded, because the two answer different questions: the raw field carries how confident the reconstruction is and the thresholded one is the defect map a caller would act on. `pr_auc` normally drops on thresholding — the ranking it integrates over is exactly what the cut throws away — while `l2_error` improves, since the grey halo around a recovered void costs more in norm than the boundary the cut misplaces

