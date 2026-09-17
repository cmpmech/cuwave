# Full Waveform Inversion

**Full waveform inversion (fwi)** recovers a material field from recorded waveforms by minimizing the misfit between measured and simulated traces, using the full recorded signal rather than picked arrival times

An inversion is set up as follows:
1. define the forward problem with [scalar](scalar.md) (here `ScalarWave`, whose indicator $\gamma$ scales the density)
2. place transducers with `line` and turn them into a shot list with `shots` and a receiver array with `Sensors` ([utils](utils.md))
3. produce or load the measurement, `measure` against the true field for a synthetic study
4. pick an [optimization](optimization.md) scheme and, if the problem needs it, a [regularization](regularization.md)
5. iterate `misfit_gradient` and one optimizer step, clipping the design back into its bounds
6. score the result against the truth with [evals](evals.md), both raw and snapped to two materials by `threshold` on the interior nodes `interior_slice` selects

$$C=\frac{1}{2}\sum_\textrm{shots}\sum_t\sum_r\left(u_r^t-u_{r,\textrm{obs}}^t\right)^2$$

with the simulated traces $u_r^t$ and the measured `observed` $u_{r,\textrm{obs}}^t$, which is `l2_misfit` and the default `objective` of `misfit_gradient`

| driver | what it adds |
|---|---|
| `examples/fwi/scalar2D_fwi_adam.py` | the reference inversion: `Adam`, no regularization, a synthetic measurement simulated at a higher order than it is inverted at |
| `examples/fwi/scalar2D_fwi_lbfgs.py` | `Lbfgs` with the Armijo line search in place of `Adam` |
| `examples/fwi/scalar2D_fwi_adam_sponge.py` | a [sponge](boundary.md) on the left and right edges, so the specimen is unbounded across the transducer array; the pad is added outside the region of interest and frozen out of the design |
| `examples/fwi/regularization/scalar2D_fwi_adam_penalty.py` | a `TotalVariation` or `Tikhonov` term added to the objective |
| `examples/fwi/regularization/scalar2D_fwi_adam_projection.py` | `DensityFilter` and `Projection` on the design, with a continuation schedule |
| `examples/fwi/regularization/scalar2D_fwi_adam_nn.py` | the design reparametrized by a [generator](nn.md) |
| `examples/fwi/scalar2D_fwi_adam_mask.py` | a mask holding the nodes around each transducer intact, so the array itself is never inverted for |
| `examples/fwi/regularization/scalar2D_fwi_lbfgs_projection.py` | the same projection under `Lbfgs`, where the map changing between iterations is what the stored curvature pairs have to survive |
| `examples/fwi/regularization/scalar2D_fwi_adam_nn_projection.py` | the generator and the projection composed, the driver fixing the order the two adjoints chain in |
| `examples/fwi/regularization/scalar2D_fwi_adam_sponge_projection.py` | the projection on the sponged domain, the layer frozen out of the design |
| `examples/fwi/regularization/scalar2D_fwi_adam_sponge_nn.py` | the generator on the sponged domain, the same freeze |
| `examples/fwi/elastic2D_fwi_adam.py` | the vector unknown: [elastic](elastic.md) in place of the pressure wave, with `direction` on every source and receiver |
| `examples/fwi/elastic2D_fwi_adam_sponge.py` | the elastic inversion with the sponge, the two extensions combined |

An inversion is scored twice, on the recovered field and on the same field thresholded, because the two answer different questions: the raw field carries how confident the reconstruction is and the thresholded one is the defect map a caller would act on. `pr_auc` normally drops on thresholding (the ranking it integrates over is exactly what the cut throws away), while `l2_error` improves, since the grey halo around a recovered void costs more in norm than the boundary the cut misplaces

