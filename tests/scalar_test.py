"""The two step kernels of the pressure family, which must be one operator.

A wide unmasked 2D step on a large grid streams along axis 0 (`USE_STREAM`): the axis-0
taps queue in registers while the row sits in shared memory. Everywhere else the step is
the per-thread kernel, so the tests force each in turn rather than trust the switch. Both take the taps in the same order, so they agree bit for
bit at order 2 and to the last bit of a contracted fma above it. What can break is the
queue across chunk boundaries, the tile halo across block boundaries and past a narrow
tile, the ghost rows the queue reads, and the damping and inertia paths, each pinned here
against the per-thread kernel.
"""

import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # cupy missing or no GPU
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.boundary import Dirichlet, sponge
    from cuwave.scalar import AcousticWave, ScalarWave
    from cuwave.signals import ricker
    from cuwave.utils import point_source
    from cuwave.wave import compile_kernels, simulate, stable_dt

NX = (150, 77)  # three chunks of axis 0, and a width no tile divides


# -------------------------------------- helpers --------------------------------------
def _scalar(order, precision="float64", boundary=None, threads=(4, 32), N=1):
    dx = (0.1, 0.1)
    return ScalarWave(
        NX,
        dx,
        N,
        0.5 * stable_dt(dx, 1.0, order),
        threads,
        precision=precision,
        space_order=order,
        wavespeed=1.0,
        density=1.0,
        boundary=boundary,
    )


def _step(sim, indicator, fields):
    """One step of `fields` = (u0, u1, u2) under the kernel `sim` compiles."""
    mat = sim.build_materials(indicator.copy())
    u0, u1, u2 = (f.copy() for f in fields)
    return sim.define_step(compile_kernels(sim), mat)(u0, u1, u2)


def _both(sim, indicator, fields):
    """The interior after one step, streamed and per thread, whatever `streams` picks."""
    steps = []
    for streams in (True, False):
        with mock.patch.object(type(sim), "streams", streams):
            steps.append(_step(sim, indicator, fields))
    interior = tuple(slice(1, n - 1) for n in sim.Nx)
    return steps[0][interior], steps[1][interior]


def _random(sim, seed):
    rng = np.random.default_rng(seed)
    indicator = cp.asarray(0.3 + rng.random(sim.Nx_padded), dtype=sim.dtype)
    fields = cp.asarray(rng.standard_normal((3, *sim.Nx_padded)), dtype=sim.dtype)
    return indicator, fields


# ---------------------------------- the two kernels ----------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class StreamingTest(unittest.TestCase):
    def _assert_same(self, sim, a, b):
        scale = float(cp.abs(b).max())
        ulps = 0.0 if sim.space_order == 2 else 8 * float(np.finfo(sim.dtype).eps)
        self.assertLessEqual(float(cp.abs(a - b).max()), ulps * scale)

    def test_the_streaming_step_is_the_per_thread_step(self):
        for order in (2, 4, 8, 16):
            for precision in ("float32", "float64"):
                for boundary in (None, Dirichlet):
                    with self.subTest(order=order, precision=precision, b=boundary):
                        sim = _scalar(order, precision, boundary)
                        self._assert_same(sim, *_both(sim, *_random(sim, order)))

    def test_a_tile_narrower_than_its_halo(self):
        # order 16 reaches 8 nodes, past a tile of 4 threads
        sim = _scalar(16, threads=(1, 4))
        self._assert_same(sim, *_both(sim, *_random(sim, 3)))

    def test_the_damped_update_and_a_formed_inertia(self):
        dx = (0.1, 0.1)
        sim = AcousticWave(
            NX,
            dx,
            1,
            1e-5,
            (4, 32),
            precision="float64",
            space_order=4,
            rho1=1.2,
            rho2=2600.0,
            kappa1=1.4e5,
            kappa2=6.9e8,
        )
        indicator, fields = _random(sim, 4)
        sim = replace(sim, damping=sponge(sim, indicator, 10, 0.1))
        self._assert_same(sim, *_both(sim, indicator, fields))

    def test_only_a_wide_unmasked_2d_step_on_a_full_grid_streams(self):
        dx = (0.1, 0.1)
        sim = ScalarWave(
            (4098, 2050),
            dx,
            1,
            1e-3,
            (4, 128),
            space_order=8,
            wavespeed=1.0,
            density=1.0,
        )
        self.assertTrue(sim.streams)
        self.assertFalse(replace(sim, space_order=4).streams)
        self.assertFalse(replace(sim, Nx=(130, 130)).streams)
        sim.domain = cp.ones(sim.Nx_padded, dtype=cp.bool_)
        self.assertFalse(sim.streams)
        sim = ScalarWave(
            (200, 200, 200),
            (0.1,) * 3,
            1,
            1e-3,
            (1, 4, 32),
            space_order=8,
            wavespeed=1.0,
            density=1.0,
        )
        self.assertFalse(sim.streams)

    def test_a_run_streams_like_the_per_thread_loop(self):
        # every step, the ghost kernels and the excitation in between
        sim = _scalar(8, N=200)
        t = np.arange(sim.N) * sim.dt
        source = point_source(sim, (3.0, 2.0), ricker(t, 1.0, 2.0))
        indicator, _ = _random(sim, 5)
        runs = []
        for streams in (True, False):
            with mock.patch.object(type(sim), "streams", streams):
                runs.append(simulate(sim, source, indicator))
        streamed, per_thread = runs
        scale = float(cp.abs(per_thread).max())
        self.assertGreater(scale, 0.0)
        self.assertLess(float(cp.abs(streamed - per_thread).max()), 1e-12 * scale)


if __name__ == "__main__":
    unittest.main()
