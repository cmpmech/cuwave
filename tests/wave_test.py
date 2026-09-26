"""`march`, the graph replay every time loop runs through.

Replaying a captured chunk has to be the plain loop exactly: same kernels, same order,
so the fields agree bit for bit. What can break is the staging around it, so each
test takes a step count that leaves a partial tail chunk and compares against
`CHUNK` patched past `N`, which forces the plain loop. The windows covered are the
three kinds: a plain input and output (`simulate`), an output with a loaded halo
(the `sensitivity` history), and reversed contiguous inputs (the strip replay of
`reconstruction_sensitivity`, with the damping that gives it a strip).
"""

import unittest
from unittest import mock

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # cupy missing or no GPU
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.boundary import sponge
    from cuwave.scalar import ScalarWave
    from cuwave.sensitivity import (
        l2_misfit,
        reconstruction_sensitivity,
        sensitivity,
    )
    from cuwave.wave import CHUNK, Source, simulate, stable_dt

NX = (40, 48)


def _setup(damped=False):
    dx = tuple(1.0 / (n - 3) for n in NX)
    N = 2 * CHUNK + 7
    sim = ScalarWave(
        NX,
        dx,
        N,
        0.5 * stable_dt(dx, 1.0, 4),
        (4, 32),
        space_order=4,
        wavespeed=1.0,
        density=1.0,
    )
    indicator = 1.0 + 0.2 * cp.random.RandomState(0).rand(*sim.Nx_padded)
    indicator = indicator.astype(sim.dtype)
    if damped:
        sim.damping = sponge(sim, indicator, 6, 0.05)
    signal = np.sin(0.3 * np.arange(N))[:, None]
    source = Source(cp.array([[20], [24]], cp.int32), cp.asarray(signal, sim.dtype))
    sensors = cp.array([[10, 10, 12], [8, 9, 30]], cp.int32)
    return sim, source, indicator, sensors


def _plain(run):
    with mock.patch("cuwave.wave.CHUNK", 10**6):
        return run()


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class MarchTest(unittest.TestCase):
    def test_simulate_replays_the_plain_loop(self):
        sim, source, indicator, sensors = _setup()
        replayed = simulate(sim, source, indicator, sensors)
        plain = _plain(lambda: simulate(sim, source, indicator, sensors))
        for a, b in zip(replayed, plain):
            np.testing.assert_array_equal(a.get(), b.get())

    def test_a_staged_history_replays_the_plain_loop(self):
        sim, source, indicator, sensors = _setup()
        objective = l2_misfit(cp.zeros((sim.N, 3), sim.dtype))
        replayed = sensitivity(sim, source, indicator, sensors, objective)
        plain = _plain(lambda: sensitivity(sim, source, indicator, sensors, objective))
        for name in replayed[1]:
            np.testing.assert_array_equal(replayed[1][name].get(), plain[1][name].get())

    def test_a_replayed_strip_matches_the_plain_loop(self):
        sim, source, indicator, sensors = _setup(damped=True)
        objective = l2_misfit(cp.zeros((sim.N, 3), sim.dtype))

        def run():
            return reconstruction_sensitivity(
                sim, source, indicator, sensors, objective
            )

        replayed, plain = run(), _plain(run)
        self.assertGreater(replayed[3]["strip"], 0)
        for name in replayed[1]:
            np.testing.assert_array_equal(replayed[1][name].get(), plain[1][name].get())


if __name__ == "__main__":
    unittest.main()
