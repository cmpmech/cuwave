# Source Inversion

**Source inversion** recovers the time signal a transducer emits from recorded waveforms, holding the material fixed: the mirror image of [fwi](fwi.md), where the material is the unknown and the source is prescribed

An inversion is set up as follows:
1. define the forward problem with [wave](wave.md), the material known and held fixed
2. place the transducer at a coordinate with `point_source` and the receivers with `Sensors` ([utils](utils.md))
3. produce or load the measurement, `measure` against the true signal for a synthetic study
4. pick an [optimization](optimization.md) scheme and a window the emission is confined to
5. iterate `source_sensitivity` and one optimizer step, `collect_source` chaining the node gradient back onto the signal of the coordinate
6. score the recovered signal against the truth with `l2_error` ([evals](evals.md))

The misfit is the one [fwi](fwi.md) minimizes, differentiated with respect to the emission rather than the material

$$C=\frac{1}{2}\sum_t\sum_r\left(u_r^t-u_{r,\textrm{obs}}^t\right)^2,\qquad\frac{\textrm{d}C}{\textrm{d}s_i^t}=W_i\,\sigma\,\lambda_i^t$$

with the emitted signal $s_i^t$ of source node $i$ at step $t$, the adjoint field $\lambda$, the cell weight $W$ and the source scaling `source_factor` $\sigma$, following [Bürchner, Schmid, Rank, Kollmannsberger & Fichtner 2026](https://doi.org/10.1016/j.ultras.2026.108221)

**The gradient is the adjoint field read at the source node.** So it is exactly `adjoint_signal` ([sensitivity](sensitivity.md)) run backwards, that helper dividing the objective derivative by $W\sigma$ on the way into the sensors and this one multiplying by $W\sigma$ on the way out of the source

| member | signature | description |
|---|---|---|
| source gradient | `source_sensitivity(sim, source, indicator, sensors, objective)` | the cost and the $(N,\,\textrm{num sources})$ derivative with respect to `source.signal` |
| coordinate chain rule | `collect_source(sim, coords, columns)` | that node gradient contracted back onto the signal of each coordinate, the transpose of `point_source` |

| driver | what it adds |
|---|---|
| `examples/fwi/source_inversion_2D_adam.py` | the reference inversion: one transducer in a homogeneous medium, `Adam` from a zero-source start, the measurement simulated at a higher order than it is inverted at |

## no forward field

The cost is **linear in the signal**, so the gradient pairs no forward field against the adjoint one and there is nothing to store: `source_sensitivity` runs in four grids whatever $N$ is, where `sensitivity` needs $N+2$ and `superposition_sensitivity` needs the reverse march to earn three. It also needs no kernel from the sensitivity module, `define_get_signal` at the source nodes being the whole of the backward accumulation

That is why the material gradient is not returned alongside. A joint source and material inversion is the natural extension and the one the source calibration of a real experiment eventually wants, but it pays for the forward history that this variant exists to avoid

## the distributed source

A transducer whose aperture is comparable to the wavelength does not emit like a point, and the effective model that reproduces its directivity is a **collection of point sources, each with its own wavelet**. Nothing in the pair above assumes one: `coords` with several rows and an $(N,\,\textrm{num})$ signal is already that parametrization, `point_source` distributing the columns and `collect_source` contracting them back

$$f(\mathbf{x},t)=\sum_i\ell_i(\mathbf{x})\,s_i(t)$$

with the multilinear node weights $\ell_i$ that `distribute` builds. The spatial resolution the reconstruction needs is a sampling question rather than a cost one: the number of source points does not change the number of simulations an iteration runs, which is one forward and one adjoint pass however many there are

## the emission window

A signal is unbounded, so the box constraint an [fwi](fwi.md) driver clips into has no analogue here. What stands in its place is the window the emission is known to fall inside, and it is not cosmetic: outside it the gradient is small but never zero, so an optimizer is free to fit reverberation with late source activity that no transducer produced

`Adam` makes that failure the default rather than a corner case, since it steps on the sign of a component and not on its size, and a direction the data barely constrains therefore takes the same step as one it constrains sharply. Masking the gradient outside the window costs one line in the driver and leaves those samples at exactly zero, the moments never leaving zero either. The alternative is a scale-aware step, which is what the [line search](optimization.md) of `Lbfgs` provides
