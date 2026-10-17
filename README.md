# CuWave

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/readme-dark.webp">
  <img src="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/readme-light.webp" width="100%">
</picture>

**CuWave** is a GPU-accelerated (single or multi), differentiable finite difference wave propagation code.
Possible applications include

<table width="100%">
<tr>
<td valign="middle"><a href="https://www.sciencedirect.com/science/article/pii/S0045782523000166"><strong>nondestructive testing via full waveform inversion</strong></a></td>
<td width="60%" align="right" valign="middle"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/fwi-dark.png">
<img width="100%" src="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/fwi-light.png" alt="nondestructive testing via full waveform inversion">
</picture></td>
</tr>
<tr>
<td valign="middle"><a href="https://doi.org/10.1007/s00158-025-04237-y"><strong>transient acoustic topology optimization</strong></a></td>
<td width="60%" align="center" valign="middle"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/tato-dark.webp">
<img width="46.512%" src="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/tato-light.webp" alt="transient acoustic topology optimization">
</picture></td>
</tr>
<tr>
<td valign="middle"><a href="https://www.science.org/doi/10.1126/sciadv.aay6946"><strong>analog neural networks</strong></a></td>
<td width="60%" align="right" valign="middle"></td>
</tr>
<tr>
<td valign="middle"><a href="https://opg.optica.org/josab/fulltext.cfm?uri=josab-38-2-496"><strong>transient photonic topology optimization</strong></a></td>
<td width="60%" align="right" valign="middle"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/tpto-dark.webp">
<img width="100%" src="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/tpto-light.webp" alt="transient photonic topology optimization">
</picture></td>
</tr>
</table>

## Documentation

- see the [documentation](https://github.com/cmpmech/cuwave/blob/main/docs/Home.md) for how the code works (AI-assisted; verify with sources for critical details)
- see [examples](https://github.com/cmpmech/cuwave/tree/main/examples) for how to apply the code

## Multiple GPUs

Several GPUs take one argument, and the driver otherwise stays as it is:
- `Simulation(..., devices=(0, 1))` splits **one grid** along its first axis over the listed GPUs, for grids too large or too slow for one. `simulate`, `sensitivity`, `reconstruction_sensitivity` and `superposition_sensitivity` then run split and return the same result as on one GPU, bitwise. Try it with the `DEVICES` setting of
	- [examples/forward/scalarND.py](https://github.com/cmpmech/cuwave/blob/main/examples/forward/scalarND.py), the forward problem in 2D or 3D
	- [examples/sensitivity/scalar2D_sensitivity.py](https://github.com/cmpmech/cuwave/blob/main/examples/sensitivity/scalar2D_sensitivity.py), the gradient
- `misfit_gradient(..., devices=[0, 1])` (and `measure`, `misfit`) deals the **shots** of an inversion out over the GPUs, which needs no communication and is the faster choice whenever one shot fits one GPU.

On two RTX 5090s the grid split raises the largest scalar 3D forward grid from 1280³ to 1600³ nodes and the largest `superposition_sensitivity` grid from 1024³ to 1280³, at 1.3 to 1.7x the speed of one GPU on large grids. See [docs/decomposition.md](https://github.com/cmpmech/cuwave/blob/main/docs/decomposition.md) for how it works and what is not split yet (`spectral_sensitivity`, `source_sensitivity`).

## Development status

**Scalar** & **acoustic wave** equations have been developed over the last 2 years and are thoroughly validated.

The **electromagnetic wave** equations (2D `ElectricWave`/`MagneticWave`, 2D/3D `DielectricWave`) were developed with AI assistance (Claude) and are cross-validated against the independent FDTD code [FDTDX](https://github.com/ymahlau/fdtdx): with the permittivity placed the same way in both codes, the probe traces agree to round-off (about 1e-15 relative, in float64) in 2D and 3D, in vacuum and through a dielectric block. Where the two codes place the permittivity differently (cuwave takes the harmonic mean onto staggered points, FDTDX staircases it), the traces differ at the discretization level (1e-4 to 1e-3), and that gap closes under refinement. See `examples/validation/` and `docs/maxwell.md`.

> [!IMPORTANT]
> The **elastic wave** equations were developed with AI assistance (Claude) and have undergone less validation. They are currently being validated against experimental results.

## Performance

CuWave's runtime for identical discretizations is comparable to that of other established wave propagation finite difference codes. Speedups in 2D with reflecting boundaries (**a ratio above 1 means CuWave is that many times faster**) compared to the following frameworks:
- **scalar wave equation**
	- [Deepwave](https://github.com/ar4/deepwave) (forward: ~1.1x, sensitivity: ~1x)
	- [NVIDIA Warp](https://github.com/NVIDIA/warp) (forward: ~2.1x, sensitivity: ~1.8x)
	- [SeismicWaves.jl](https://github.com/GinvLab/SeismicWaves.jl) (forward: ~4.4x, sensitivity: ~4x)
	- [Devito](https://github.com/devitocodes/devito) on CPU (forward: ~3.6x, sensitivity: ~3.8x)
- **elastic wave equation**
	- Deepwave (forward: ~1.1x, sensitivity: ~1.3x)
	- SeismicWaves.jl (forward: ~4x, sensitivity: ~5x)
Tested on one NVIDIA RTX PRO 500 Blackwell laptop GPU (6 GB) on the largest possible grids with CuWave's `superposition_sensitivity` as reference for the sensitivities. The specific numbers need to be taken with a grain of salt, as they are subject to specific hardware and simulation setup. All implementations operate on the same order of magnitude.

Additional benefits of **CuWave** are
- the built-in **higher order finite difference** schemes, allowing for fewer grid points
- a sensitivity analysis whose **memory is independent of the number of timesteps**, allowing for orders of magnitude larger grids

## Install

```bash
pip install cuwave
```

Requires an NVIDIA GPU and [CuPy](https://docs.cupy.dev/en/stable/install.html) matching your CUDA toolkit (e.g. `pip install cupy-cuda12x`), which is not pulled in automatically. For the full installation, including the optional PyTorch and running the tests, see [docs/install.md](https://github.com/cmpmech/cuwave/blob/main/docs/install.md).

## References

If you use our code for your scientific research, please acknowledge this by referring to the following publication:

_Herrmann, L., Bürchner, T., Kudela, L., Kollmannsberger, S., 2026, **A memory-efficient adjoint method to enable billion parameter optimization on a single GPU in dynamic problems**, Structural and Multidisciplinary Optimization, Volume 69, 52 (2026), DOI: [10.1007/s00158-025-04237-y](https://doi.org/10.1007/s00158-025-04237-y)_

## Acknowledgements

Ideas adapted from [FDTDX](https://github.com/ymahlau/fdtdx) ([Mahlau, Schubert, Berg & Rosenhahn 2026](https://doi.org/10.21105/joss.08912)), which also serves as the reference for validating the electromagnetic solvers:
- `spectral_sensitivity`, the frequency-domain adjoint for spectral objectives, following FDTDX's frequency-mode adjoint and the hybrid time/frequency adjoint of Meep ([Hammond, Oskooi, Johnson & Ralph 2022](https://doi.org/10.1364/OE.442074))
- `SmoothedProjection`, the subpixel-smoothed projection of [Hammond et al. 2025](https://arxiv.org/abs/2503.20189), as FDTDX implements it
- the reduced-precision boundary record behind `reconstruction_sensitivity(strip_dtype=...)`

The harmonic permittivity mean onto each `DielectricWave` component point, adopted after the comparison with FDTDX, follows the subpixel smoothing of [Farjadpour et al. 2006](https://doi.org/10.1364/OL.31.002972).

No code was copied: FDTDX differentiates with JAX, and every adjoint here is derived by hand.

## Contact

For questions, bug reports, or collaboration inquiries, please don't hesitate to contact Leon Herrmann at [herrmann.leon@pm.me](mailto:herrmann.leon@pm.me).

## License

MIT; see [LICENSE](https://github.com/cmpmech/cuwave/blob/main/LICENSE).
