"""The array module the package computes with, chosen once at import.

cupy where it finds a CUDA device, else `cpu`: numpy, with the kernels compiled for the
host. `CUWAVE_DEVICE=cpu` takes the host even beside a GPU, `CUWAVE_DEVICE=gpu` insists
on the GPU. Every module imports `cp`, `cpt` and `ndi` from here, never cupy itself.
"""

import os

DEVICE = os.environ.get("CUWAVE_DEVICE", "")
if DEVICE not in ("", "cpu", "gpu"):
    raise ValueError(f"CUWAVE_DEVICE {DEVICE!r} is neither cpu nor gpu")


def _finds_gpu() -> bool:
    """Whether cupy imports and sees a CUDA device."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # cupy missing, or no driver
        return False


ON_GPU = DEVICE == "gpu" or (DEVICE == "" and _finds_gpu())

if ON_GPU:
    import cupy as cp
    import cupy.typing as cpt
    import cupyx.scipy.ndimage as ndi
else:  # the names the package imports from here
    import numpy.typing as cpt  # noqa: F401
    import scipy.ndimage as ndi  # noqa: F401

    from . import cpu as cp  # noqa: F401
