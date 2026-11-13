# Install

**Installation** is kept lightweight: only CuPy is required beyond the standard Python library, and it is installed separately because its wheel depends on the CUDA toolkit on the machine

```bash
pip install cupy-cuda12x        # or cupy-cuda11x, to match your CUDA
pip install cuwave               # or `pip install -e .` from a checkout
```

All remaining dependencies are declared in `pyproject.toml` and come with the wheel

## CPU

Without CuPy or a CUDA device the package runs on the CPU, compiling the kernels with the system C++ compiler and OpenMP (see [cpu](cpu.md)); `CUWAVE_DEVICE=cpu` selects it beside a GPU too. The `cpu` extra adds SciPy:

```bash
pip install "cuwave[cpu]"
```

The compiler is `CXX` (default `c++`), with the flags `CUWAVE_CXXFLAGS` (default `-O3 -march=native -fopenmp -ffp-contract=off`). Linux's GCC works as installed; Apple Clang needs `libomp` from Homebrew and the flags it names (or Homebrew's GCC as `CXX`); on Windows use WSL2

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
python -m pytest tests/ -q  # ~20 s on a GPU; without one the suite runs on the CPU, the GPU-only tests and PyTorch's skip
```
