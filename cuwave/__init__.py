"""CuWave: a single-GPU, differentiable finite-difference wave solver"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cuwave")
except PackageNotFoundError:  # a checkout on sys.path without an install
    __version__ = "0.0.0"
