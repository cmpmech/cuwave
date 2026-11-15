"""The host compiler against cupy's launch contract, and the CPU solver against the GPU one.

`cpu.RawModule` stands in for `cupy.RawModule`, so `HostKernelTest` pins what the package
relies on: every thread of a launch runs once with its own indices, a scalar arrives by
value in its dtype and an array by pointer, a `-D` flag defines, a `__constant__` table
written through `get_global` reaches the kernel, `atomicAdd` loses no update, and what the
host cannot run (shared memory, a strided array, a flag it does not know) raises rather
than computing garbage. These need a C++ compiler, not a GPU.

`AgreementTest` runs `sensitivity` for each equation on the GPU and in a subprocess under
`CUWAVE_DEVICE=cpu`, on the same float64 inputs, and holds the record and the gradients
to round-off: the forward, adjoint, gradient, wall and pml kernels of every equation as
the host compiles them. A GPU machine otherwise runs the whole suite on the GPU, so this
is the only check of the CPU path there.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cuwave import cpu
from cuwave.anisotropic import AnisotropicElasticWave
from cuwave.backend import ON_GPU, cp
from cuwave.boundary import Dirichlet, Neumann, Pml
from cuwave.elastic import ElasticWave
from cuwave.maxwell import DielectricWave
from cuwave.scalar import AcousticWave, ScalarWave
from cuwave.sensitivity import l2_misfit, sensitivity
from cuwave.signals import ricker
from cuwave.utils import point_source
from cuwave.wave import stable_dt

KERNELS = Path(__file__).with_suffix(".cu")
HAS_COMPILER = shutil.which(os.environ.get("CXX", "c++")) is not None
EQUATIONS = ("scalar", "scalar3D", "acoustic", "elastic", "maxwell", "anisotropic")
RTOL = 1e-12  # float64 round-off: the fields agree to about 1e-14


# -------------------------------------- helpers --------------------------------------
def _module(*options):
    return cpu.RawModule(KERNELS.read_text(), options)


def _sim(name):
    """A small float64 simulation of equation `name`, walls and a pml mixed in."""
    two = dict(Nx=(40, 36), threads=(8, 8))
    if name == "scalar":
        mixed = ((Neumann, Dirichlet), (Dirichlet, Neumann))
        return _build(ScalarWave, 4, **two, boundary=mixed, wavespeed=1.0, density=1.0)
    if name == "scalar3D":
        three = dict(Nx=(20, 18, 24), threads=(4, 4, 8))
        return _build(ScalarWave, 2, **three, wavespeed=1.0, density=1.0)
    if name == "acoustic":
        return _build(
            AcousticWave,
            4,
            **two,
            pml=Pml(6, 1.0),
            rho1=1.0,
            rho2=2.0,
            kappa1=1.0,
            kappa2=3.0,
        )
    common = dict(density=1.0, wavespeed_p=1.0, wavespeed_s=0.5)
    if name == "elastic":
        return _build(ElasticWave, 4, **two, **common)
    if name == "anisotropic":
        return _build(AnisotropicElasticWave, 2, **two, **common)
    return _build(DielectricWave, 4, **two, permittivity1=1.0, permittivity2=4.0)


def _build(cls, order, Nx, threads, **kwargs):
    dx = tuple(1.0 / (n - 3) for n in Nx)
    dt = 0.5 * stable_dt(dx, 2.0, order)
    return cls(
        Nx, dx, 60, dt, threads, precision="float64", space_order=order, **kwargs
    )


def _run(name):
    """(record, gradients) of `sensitivity` on equation `name`, as host arrays."""
    sim = _sim(name)
    rng = np.random.default_rng(EQUATIONS.index(name))
    t = np.arange(sim.N) * sim.dt
    middle = [(n // 2 - 3) * h for n, h in zip(sim.Nx, sim.dx)]
    direction = None if sim.ncomp == 1 else [0.6, 0.8] + [0.0] * (sim.ncomp - 2)
    source = point_source(sim, [middle], ricker(t, 1.0, 8.0), direction=direction)
    nodes = [[n // 2 + 3 for n in sim.Nx], [n // 3 for n in sim.Nx]]
    rows = [[c, *node] for node in nodes for c in range(sim.ncomp)]
    rows = rows if sim.ncomp > 1 else nodes
    sensors = cp.asarray(np.array(rows).T, dtype=cp.int32)
    indicator = cp.asarray(0.6 + 0.3 * rng.random(sim.Nx_padded), dtype=sim.dtype)
    observed = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)
    _, grads, traces, _ = sensitivity(
        sim, source, indicator, sensors, l2_misfit(observed)
    )
    return {"traces": cp.asnumpy(traces)} | {k: cp.asnumpy(v) for k, v in grads.items()}


def _dump(path):
    """Every equation's `_run`, saved to `path`; the subprocess calls it on the CPU."""
    results = {f"{n}/{k}": v for n in EQUATIONS for k, v in _run(n).items()}
    np.savez(path, **results)


# ----------------------------------- host launches -----------------------------------
@unittest.skipUnless(HAS_COMPILER, "needs a C++ compiler")
class HostKernelTest(unittest.TestCase):
    def test_every_thread_runs_once_with_its_own_indices(self):
        grid, block = (3, 2, 2), (4, 3, 2)
        n = int(np.prod(grid) * np.prod(block))
        for options, offset in (((), 0), (("-DOFFSET=5",), 5)):
            with self.subTest(options=options):
                out = np.full(n, -1, dtype=np.int32)
                kernel = _module(*options).get_function("index_kernel")
                kernel(grid, block, (out, np.int32(n)))
                np.testing.assert_array_equal(out, np.arange(n) + offset)

    def test_a_scalar_arrives_by_value_in_its_dtype(self):
        out = np.zeros(4)
        args = (out, np.int32(-7), np.float32(0.1), 0.3, 2**40)
        _module().get_function("scalar_kernel")((1,), (1,), args)
        np.testing.assert_array_equal(out, [-7.0, np.float32(0.1), 0.3, 2.0**40])

    def test_a_constant_table_written_from_the_host_reaches_the_kernel(self):
        module = _module()
        np.ndarray((4,), np.float64, module.get_global("TABLE"))[...] = [1, 2, 3, 4]
        out = np.zeros(4)
        module.get_function("table_kernel")((1,), (4,), (out,))
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])

    def test_atomic_add_loses_no_update(self):
        total = np.zeros(1, dtype=np.int32)
        _module().get_function("count_kernel")((64,), (64,), (total,))
        self.assertEqual(int(total[0]), 64 * 64)

    def test_a_kernel_the_flags_drop_is_not_named(self):
        with self.assertRaises(ValueError):
            _module().get_function("shared_kernel")

    def test_what_the_host_cannot_run_raises(self):
        out = np.zeros(8, dtype=np.int32)
        kernel = _module().get_function("index_kernel")
        with self.assertRaises(RuntimeError):
            _module("-DSHARED")  # shared memory does not compile
        with self.assertRaises(ValueError):
            kernel((1,), (4,), (out[::2], np.int32(4)))
        with self.assertRaises(ValueError):
            kernel((1,), (4,), (out, np.int32(4)), shared_mem=16)
        with self.assertRaises(ValueError):
            _module("-arch=sm_80")


# ------------------------------ the CPU against the GPU ------------------------------
@unittest.skipUnless(ON_GPU and HAS_COMPILER, "needs a CUDA device and a C++ compiler")
class AgreementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpu.npz"
            script = f"import cpu_test; cpu_test._dump({str(path)!r})"
            env = os.environ | {"CUWAVE_DEVICE": "cpu"}
            command = [sys.executable, "-c", script]
            here = str(Path(__file__).parent)
            done = subprocess.run(command, cwd=here, env=env, capture_output=True)
            if done.returncode:
                raise RuntimeError(f"the CPU run failed:\n{done.stderr.decode()}")
            with np.load(path) as host:
                cls.host = dict(host)

    def test_every_equation_matches_the_gpu_to_round_off(self):
        for name in EQUATIONS:
            for key, device in _run(name).items():
                with self.subTest(equation=name, field=key):
                    host = self.host[f"{name}/{key}"]
                    scale = np.abs(device).max()  # 0 for maxwell's stiffness, held exactly
                    self.assertLessEqual(np.abs(host - device).max(), RTOL * scale)
            self.assertGreater(np.abs(self.host[f"{name}/traces"]).max(), 0.0)
