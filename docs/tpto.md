# Transient Photonic Topology Optimization

**Transient photonic topology optimization (tpto)** distributes silicon inside a design region so that a device does something optical with the light crossing it: focus it, or route it by wavelength. It is the [tato](tato.md) recipe with [maxwell](maxwell.md) in place of the acoustic wave equation and a spectral figure of merit in place of a time integral, following the tutorial of [Christiansen & Sigmund 2021](https://doi.org/10.1364/JOSAB.406048) and the review of [Jensen & Sigmund 2010](https://doi.org/10.1002/lpor.201000014)

An optimization is set up as follows:
1. define the forward problem with [maxwell](maxwell.md), `ElectricWave` or `MagneticWave` depending on the polarization, its indicator $\gamma$ interpolating between air and silicon
2. open the domain with `pad_for_sponge` and `sponge` ([boundary](boundary.md)), which for Maxwell is a real conductivity rather than an absorbing fiction
3. mark the design region with `box` and the ports with `nodes` ([geometry](geometry.md))
4. build the objective with `intensity` ([utils](utils.md)), one broadband `ricker` run scoring every design wavelength at once
5. map the design variables through a `DensityFilter` and a `Projection` ([regularization](regularization.md))
6. iterate `response_gradient` with `reconstruction_sensitivity` and one `Adam` step, sharpening the projection on a continuation schedule
7. snap the final design to 0/1 with `threshold` and re-simulate it to report what it achieves

## the figure of merit

The quantity a photonic device is judged on is time-harmonic, so the objective is a discrete Fourier transform of the sensor record and not an integral over it

$$\Phi=\sum_f\sum_c w_{fc}\left|\hat{u}_{fc}\right|^2,\qquad\hat{u}_{fc}=\Delta t\sum_n e^{-i2\pi f_nt_n}u_{nc}$$

with the design frequencies $f$, the sensor nodes $c$, and the weights $w$ that pair each frequency with the port it is scored on. Nothing new is needed to differentiate it: an objective already receives the whole $(N,\,\textrm{num sensors})$ record and returns its derivative, so the transform is a matrix on the host side of the adjoint and `intensity` in [utils](utils.md) is the whole of it

**One broadband run scores every wavelength.** The tutorial solves one state equation per design wavelength, six of them for its robust demultiplexer; a transient solve carries the whole band in a single `ricker` pulse and reads each frequency off the same record. A second wavelength therefore costs one row of the weight matrix rather than another simulation, which is the one structural advantage the time domain has here

The run has to be long enough that the transform resolves the design frequencies: the bin spacing is $1/\left(N\Delta t\right)$, so a demultiplexer separating two wavelengths by $\Delta f$ needs $N\Delta t\gg1/\Delta f$, on top of the time the device takes to ring down

## the two drivers

`examples/tpto/metalens2D_optimization.py` reproduces the cylindrical metalens of the tutorial's case 3: a `MagneticWave` slab of silicon and air focusing a Gaussian-enveloped normally incident beam into a point at numerical aperture 0.9. Its objective is `intensity` at the focal point, maximized, and the gain is quoted against a run with the lens region emptied, so the source calibration and the layer losses drop out. Passing three frequencies instead of one is the tutorial's broadband case 4, at no extra cost

`examples/tpto/demultiplexer2D_optimization.py` reproduces the wavelength demultiplexer: an `ElectricWave` design region routing 1300 nm into one output waveguide and 1550 nm into the other. Its objective is one `intensity` with a two-row weight matrix, each row selecting one wavelength on one port. What it reports is the $2\times2$ table of port against wavelength, since the isolation between the two rows is the thing the device is for and a total transmittance is not

Both are nondimensional: the design wavelength, the vacuum speed, the background permittivity and the permeability are all 1, so a length is a number of wavelengths and the frequency of interest is 1. SI-scale permittivities in float32 would put the gradient near the underflow of Adam's second moment

## thresholded evaluation

**The design the optimizer converges to is grey and no structure achieves its objective**, exactly as in [tato](tato.md): an intermediate $\gamma$ has an intermediate permittivity that no mixture of air and silicon has, and only the thresholded design can be etched. Both drivers snap the result at $\eta$ and re-simulate it, reporting the discretization price next to `non_discreteness` ([evals](evals.md))

The tutorial reaches a discrete design a second way, by adding an imaginary part $-i\alpha\gamma\left(1-\gamma\right)$ to the interpolated permittivity so that grey material absorbs. That is a **design-dependent conductivity**, and `Simulation.damping` is a constructor field the gradient does not run through, so it is out of scope here. The filter and projection continuation of the tutorial's case 3 is the route taken instead, and it is the one the tutorial itself prefers, since explicit penalization tends to stick in poor local minima

## what the layer costs

The sponge is a lossy layer and not a perfectly matched one. Around $-28$ dB of amplitude reflection is what two wavelengths of it buy at a well-tuned `beta`, and that is against a plane wave: a **guided mode running into the layer** is terminated considerably worse, which is why the demultiplexer lines its faces three wavelengths deep and runs its waveguides on through. This is the largest single obstacle to matching the tutorial's transmittances, larger than anything about the discretization, and a perfectly matched layer is the fix rather than a thicker sponge

Because the layer is lossy, the domain is not reversible and `superposition_sensitivity` refuses it, so both drivers use `reconstruction_sensitivity` ([sensitivity](sensitivity.md)) and print the `drift` its reverse march reports. A full history would be several gigabytes at these step counts, where the strip is tens of megabytes
