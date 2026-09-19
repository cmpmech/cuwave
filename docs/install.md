# Install

**Installation** is kept lightweight: only CuPy is required beyond the standard Python library, and it is installed separately because its wheel depends on the CUDA toolkit on the machine

```bash
pip install cupy-cuda12x        # or cupy-cuda11x, to match your CUDA
pip install cuwave               # or `pip install -e .` from a checkout
```

All remaining dependencies are declared in `pyproject.toml` and come with the wheel

## PyTorch

PyTorch is optional and only needed for the regularization via neural optimization ([neural networks](nn.md)); see [pytorch](https://pytorch.org/get-started/locally/) for the installation

> [!NOTE]
> Match PyTorch's CUDA version to CuPy's, or the two runtimes clash at the first kernel launch.
> With `cupy-cuda12x`:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```

## Tests

The tests under `tests/` are `unittest` classes, but `pytest` is the recommended runner:

```bash
pip install pytest
python -m pytest tests/ -q  # ~20 s on a GPU, ~3 s without: CUDA and PyTorch tests skip when unavailable
```
