# CuWave

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/readme-dark.webp">
  <img src="https://raw.githubusercontent.com/cmpmech/cuwave/main/.assets/readme-light.webp" width="100%">
</picture>

**CuWave** is a single-GPU, differentiable finite difference wave propagation code.
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

## Development status

**Scalar** & **acoustic wave** equations have been developed over the last 2 years and are thoroughly validated.

> [!IMPORTANT]
> **Elastic** & **electromagnetic wave** equations were developed with AI assistance (Claude) and have undergone less validation. The elastic wave equation is currently being validated against experimental results.

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

## Contact

For questions, bug reports, or collaboration inquiries, please don't hesitate to contact Leon Herrmann at [herrmann.leon@pm.me](mailto:herrmann.leon@pm.me).

## License

MIT; see [LICENSE](https://github.com/cmpmech/cuwave/blob/main/LICENSE).
