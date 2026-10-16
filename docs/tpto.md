# Transient Photonic Topology Optimization

**Transient photonic topology optimization (tpto)** distributes silicon inside a design region so that a device does something optical with the light crossing it: focus it, or route it by wavelength. It is the [tato](tato.md) recipe with [maxwell](maxwell.md) in place of the acoustic wave equation and a spectral figure of merit in place of a time integral, following the tutorial of [Christiansen & Sigmund 2021](https://doi.org/10.1364/JOSAB.406048) and the review of [Jensen & Sigmund 2010](https://doi.org/10.1002/lpor.201000014)

An optimization is set up as follows:
1. define the forward problem with [maxwell](maxwell.md), `ElectricWave` or `MagneticWave` depending on the polarization, its indicator $\gamma$ interpolating between air and silicon
2. open the domain with `pad_for_sponge` and a `Pml` ([boundary](boundary.md)), or a `sponge`, which for Maxwell is a real conductivity rather than an absorbing fiction
3. mark the design region with `box` and the ports with `nodes` ([geometry](geometry.md))
4. build the objective with `intensity` ([utils](utils.md)), one broadband `ricker` run scoring every design wavelength at once
5. map the design variables through a `DensityFilter` and a `Projection` ([regularization](regularization.md))
6. iterate `response_gradient` with `spectral_sensitivity` on the design region and one `Adam` step, sharpening the projection on a continuation schedule
7. snap the final design to 0/1 with `threshold` and re-simulate it to report what it achieves

## the figure of merit

The quantity a photonic device is judged on is time-harmonic, so the objective is a discrete Fourier transform of the sensor record and not an integral over it

$$\Phi=\sum_f\sum_c w_{fc}\left|\hat{u}_{fc}\right|^2,\qquad\hat{u}_{fc}=\Delta t\sum_n e^{-i2\pi f_nt_n}u_{nc}$$

with the design frequencies $f$, the sensor nodes $c$, and the weights $w$ that pair each frequency with the port it is scored on. Nothing new is needed to differentiate it: an objective already receives the whole $(N,\,\textrm{num sensors})$ record and returns its derivative, so the transform is a matrix on the host side of the adjoint and `intensity` in [utils](utils.md) is the whole of it

**One broadband run scores every wavelength.** The tutorial solves one state equation per design wavelength, six of them for its robust demultiplexer; a transient solve carries the whole band in a single `ricker` pulse and reads each frequency off the same record. A second wavelength therefore costs one row of the weight matrix rather than another simulation, which is the one structural advantage the time domain has here

The run has to be long enough that the transform resolves the design frequencies: the bin spacing is $1/\left(N\Delta t\right)$, so a demultiplexer separating two wavelengths by $\Delta f$ needs $N\Delta t\gg1/\Delta f$, on top of the time the device takes to ring down

## the driver

`examples/tpto/metalens2D_optimization.py` reproduces the cylindrical metalens of the tutorial's case 3: a `MagneticWave` slab of silicon and air focusing a Gaussian-enveloped normally incident beam into a point at numerical aperture 0.9. Its objective is `intensity` at the focal point, maximized, and the gain is quoted against a run with the lens region emptied, so the source calibration and the layer losses drop out. Passing three frequencies instead of one is the tutorial's broadband case 4, at no extra cost

The driver is nondimensional: the design wavelength, the vacuum speed, the background permittivity and the permeability are all 1, so a length is a number of wavelengths and the frequency of interest is 1. SI-scale permittivities in float32 would put the gradient near the underflow of Adam's second moment

## thresholded evaluation

**The design the optimizer converges to is grey and no structure achieves its objective**, exactly as in [tato](tato.md): an intermediate $\gamma$ has an intermediate permittivity that no mixture of air and silicon has, and only the thresholded design can be etched. The driver snaps the result at $\eta$ and re-simulates it, reporting the discretization price next to `non_discreteness` ([evals](evals.md))

The tutorial reaches a discrete design a second way, by adding an imaginary part $-i\alpha\gamma\left(1-\gamma\right)$ to the interpolated permittivity so that grey material absorbs. That is a **design-dependent conductivity**, and `Simulation.damping` is a constructor field the gradient does not run through, so it is out of scope here. The filter and projection continuation of the tutorial's case 3 is the route taken instead, and it is the one the tutorial itself prefers, since explicit penalization tends to stick in poor local minima

## what the layer costs

The driver lines every face with a half-wavelength `Pml`. A sponge is a lossy layer and not a perfectly matched one: two wavelengths of it at a well-tuned `beta` still return a few percent of a plane wave, and a **guided mode running into it** is terminated considerably worse, which is the largest single obstacle to matching the tutorial's transmittances. The matched layer leaves less behind at a quarter of that thickness and terminates a guide as well as a plane wave, as long as the guide runs on through it rather than stopping at the boundary, and the thinner layer is grid the run no longer carries

Because the layer absorbs, the domain is not reversible and `superposition_sensitivity` refuses it. The cost is spectral, though, so the driver uses `spectral_sensitivity` ([sensitivity](sensitivity.md)), which keeps the DFT of both runs over the design region instead of any history and prints the `decay` that says how far they settled; `reconstruction_sensitivity` is the drop-in alternative for a cost that is not spectral, at the price of a reverse march and a strip. The run is longer than the transform alone needs, since a resonant lens rings past its pulse and the warning past `DECAY_LIMIT` is the cue to lengthen it
