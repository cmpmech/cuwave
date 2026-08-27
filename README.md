<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".assets/readme-dark.png">
  <img src=".assets/readme-light.png" width="100%">
</picture>

**CuWave** is a single GPU-accelerated differentiable finite difference wave propagation code.
Possible applications include

- [**nondestructive testing via full waveform inversion**](https://www.sciencedirect.com/science/article/pii/S0045782523000166)
- <picture>
  <source media="(prefers-color-scheme: dark)" srcset=".assets/tato-dark.png">
  <img align="right" width="25%" src=".assets/tato-light.png" alt="transient acoustic topology optimization">
  </picture>
  <a href="https://doi.org/10.1007/s00158-025-04237-y"><strong>transient acoustic topology optimization</strong></a>
  <a href="https://doi.org/10.1007/s00158-025-04237-y" tabindex="-1"><img align="middle" width="6.25%" src=".assets/spacer.png" alt=""></a>
- [**analog neural networks**](https://www.science.org/doi/10.1126/sciadv.aay6946)

<br clear="right">

## Documentation
- see the [documentation](docs/Home.md) for how the code works
- see [examples](https://github.com/Leon-Herrmann/cuwave/tree/main/examples) for how to apply the code
## Install

```bash
pip install cupy-cuda12x        # or cupy-cuda11x, to match your CUDA
pip install -e .                
```

CuPy must be installed separately because the wheel depends on your CUDA toolkit;
all remaining dependencies are declared in `pyproject.toml`.

PyTorch is optional for the regularization via neural optimization; see [pytorch](https://pytorch.org/get-started/locally/) for the installation. Otherwise it is not needed.
## References

If you use our code for your scientific research, please acknowledge this by referring to the following publication:

_Herrmann, L., Bürchner, T., Kudela, L., Kollmannsberger, S., 2026, **A memory-efficient adjoint method to enable billion parameter optimization on a single GPU in dynamic problems**, Structural and Multidisciplinary Optimization, Volume 69, 52 (2026), DOI: [10.1007/s00158-025-04237-y](https://doi.org/10.1007/s00158-025-04237-y)_
