"""numpy in the role of cupy: what the package calls beyond numpy, and kernels for the host.

A name not defined here is numpy's. The host is one synchronous device, so streams and
events are no-ops. `RawModule` compiles the CUDA source with the host C++ compiler and
OpenMP behind `kernels/cpu.hpp`, and caches the library on disk by source and flags.
"""

import ctypes
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

HEADER = Path(__file__).parent / "kernels" / "cpu.hpp"
CXXFLAGS = "-O3 -march=native -fopenmp -ffp-contract=off"  # CUWAVE_CXXFLAGS default
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cuwave"
SUFFIX = ".dll" if sys.platform == "win32" else ".so"


def __getattr__(name: str):
    """numpy's `name`: the array calls the package makes alike on cupy and numpy."""
    return getattr(np, name)


def asnumpy(a: npt.ArrayLike) -> np.ndarray:
    """A copy of `a`, as `cupy.asnumpy` copies a device array, so a later step leaves it."""
    return np.array(a)


def get_array_module(*args) -> object:
    """numpy, whatever `args` hold."""
    return np


# ------------------------------------ CUDA runtime -----------------------------------
class Stream:
    """A stream of the synchronous host: entering, waiting and synchronizing do nothing."""

    def __init__(self, non_blocking: bool = False) -> None:
        pass

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def synchronize(self) -> None:
        pass

    def wait_event(self, event: "Event") -> None:
        pass


class Event:
    """An event of the synchronous host: always reached."""

    def __init__(self, disable_timing: bool = False) -> None:
        pass

    def record(self, stream: Stream | None = None) -> None:
        pass

    def synchronize(self) -> None:
        pass


Stream.null = Stream()
cuda = SimpleNamespace(
    Stream=Stream,
    Event=Event,
    get_current_stream=lambda: Stream.null,
    memory=SimpleNamespace(OutOfMemoryError=MemoryError),
)


# -------------------------------------- kernels --------------------------------------
class RawKernel:
    """A compiled kernel, launched as cupy launches one: `kernel(grid, block, args)`."""

    def __init__(self, launch: ctypes._CFuncPtr) -> None:
        launch.restype = None
        self._launch = launch

    def __call__(
        self,
        grid: int | tuple[int, ...],
        block: int | tuple[int, ...],
        args: list | tuple,
        shared_mem: int = 0,
    ) -> None:
        """Launch over `grid` blocks of `block` threads; an array is passed by pointer, a scalar by value as cupy passes it."""
        if shared_mem:
            raise ValueError(f"shared_mem {shared_mem}: a host kernel shares no memory")
        held = [_argument(a) for a in args]
        pointers = (ctypes.c_void_p * len(held))(*(address for _, address in held))
        self._launch(*_extent(grid), *_extent(block), pointers)


class RawModule:
    """`cupy.RawModule` for the host: `code` compiled with the `-D` `options`.

    `--use_fast_math` is accepted and dropped, so the host keeps IEEE arithmetic. The
    compiler is `CXX` (default `c++`) with `CUWAVE_CXXFLAGS` (default `CXXFLAGS`); it
    needs C++17 and OpenMP. A kernel that shares memory within a block or synchronizes
    it does not compile.
    """

    def __init__(self, code: str, options: tuple[str, ...] = ()) -> None:
        unknown = [o for o in options if not o.startswith("-D")]
        if set(unknown) - {"--use_fast_math"}:
            raise ValueError(f"options {unknown} have no host equivalent")
        defines = [o for o in options if o.startswith("-D")]
        self._library = _compile(HEADER.read_text() + code, defines)

    def get_function(self, name: str) -> RawKernel:
        """The kernel `name`."""
        try:
            return RawKernel(getattr(self._library, f"launch_{name}"))
        except AttributeError:
            raise ValueError(f"the module defines no kernel {name!r}") from None

    def get_global(self, name: str) -> ctypes.Array:
        """The memory of the module global `name` (a `__constant__` table), writable."""
        try:
            size = ctypes.c_ulong.in_dll(self._library, f"sizeof_{name}").value
        except ValueError:
            raise ValueError(f"the module defines no global {name!r}") from None
        return (ctypes.c_char * size).in_dll(self._library, name)


# -------------------------------------- helpers --------------------------------------
def _argument(a) -> tuple[object, int]:
    """(what keeps it alive, its address) of kernel argument `a`, an array by pointer."""
    if isinstance(a, np.ndarray):
        if not a.flags.c_contiguous:
            raise ValueError(
                f"kernel argument of strides {a.strides} is not C-contiguous"
            )
        pointer = ctypes.c_void_p(a.ctypes.data)
        return pointer, ctypes.addressof(pointer)
    if a is None:
        pointer = ctypes.c_void_p(None)
        return pointer, ctypes.addressof(pointer)
    value = np.asarray(a)  # a Python int as int64 and a float as float64, as cupy
    return value, value.ctypes.data


def _extent(dims: int | tuple[int, ...]) -> list[int]:
    """`dims` as the three of a CUDA `dim3`, x first."""
    dims = [int(dims)] if np.isscalar(dims) else [int(d) for d in dims]
    return dims + [1] * (3 - len(dims))


def _run(command: list[str], source: str = "") -> str:
    """The output of `command` fed `source`, raising with the compiler's message on failure."""
    try:
        done = subprocess.run(command, input=source, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"no C++ compiler {command[0]!r}: install one with OpenMP, or name it in CXX"
        ) from None
    if done.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{done.stderr}")
    return done.stdout


def _compile(source: str, defines: list[str]) -> ctypes.CDLL:
    """`source` built into a shared library, from the cache if built before."""
    compiler = os.environ.get("CXX", "c++")
    flags = os.environ.get("CUWAVE_CXXFLAGS", CXXFLAGS).split()
    flags += ["-std=c++17", "-shared", "-fPIC", "-w", *defines]
    key = hashlib.sha256("\0".join([compiler, *flags, source]).encode()).hexdigest()
    library = CACHE / f"{key[:32]}{SUFFIX}"
    if not library.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        # the launchers of what the preprocessor keeps, so a kernel under #if is not named
        scan = _run(
            [compiler, *flags, "-DCUWAVE_SCAN", "-E", "-P", "-x", "c++", "-"], source
        )
        kernels = re.findall(r"\bCUWAVE_KERNEL\s+void\s+(\w+)\s*\(", scan)
        tables = re.findall(r"\bCUWAVE_GLOBAL\s+[\w\s]*?(\w+)\s*\[", scan)
        exports = [f"CUWAVE_LAUNCH({k})\n" for k in kernels]
        exports += [f"CUWAVE_SIZEOF({t})\n" for t in tables]
        with tempfile.TemporaryDirectory(dir=CACHE) as build:
            path = Path(build) / "module.cpp"
            path.write_text(source + "".join(exports))
            built = Path(build) / f"module{SUFFIX}"
            _run([compiler, *flags, str(path), "-o", str(built)])
            os.replace(built, library)  # atomic, as another process may build it too
    return ctypes.CDLL(str(library))
