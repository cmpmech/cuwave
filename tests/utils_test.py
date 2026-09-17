"""Contracts of the spectral objective, which nothing downstream can check for itself.

`intensity` is a linear transform of the sensor record followed by a squared magnitude,
and both halves are easy to get subtly wrong: the sign and the `dt` of the transform
decide whether the reported spectrum is the physical one, and its adjoint is one
conjugation away from an expression that agrees on real data and disagrees on
everything else. So the transform is pinned against `numpy.fft` and its derivative
against central differences.
"""

import unittest

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.scalar import ScalarWave
    from cuwave.utils import intensity


# -------------------------------------- helpers --------------------------------------
def _sim(N=64, dt=0.013):
    return ScalarWave(
        (32, 32),
        (0.1, 0.1),
        N,
        dt,
        (8, 8),
        precision="float64",
        wavespeed=1.0,
        density=1.0,
    )


def _record(sim, columns=4, seed=0):
    values = np.random.default_rng(seed).standard_normal((sim.N, columns))
    return cp.asarray(values, dtype=sim.dtype)


def _transform(sim, frequencies, record):
    """The transform `intensity` is built on, evaluated on the host."""
    t = np.arange(sim.N) * sim.dt
    return sim.dt * np.exp(-2j * np.pi * np.outer(frequencies, t)) @ record.get()


# ----------------------------------- the objective -----------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class IntensityTest(unittest.TestCase):
    frequencies = (0.7, 2.3, 5.0)

    def test_the_transform_matches_the_discrete_fourier_transform(self):
        sim = _sim()
        record = _record(sim)
        cost, _ = intensity(sim, self.frequencies)(record)
        reference = np.sum(np.abs(_transform(sim, self.frequencies, record)) ** 2)
        self.assertAlmostEqual(cost / reference, 1.0, places=10)

    def test_one_frequency_needs_no_sequence(self):
        sim = _sim()
        record = _record(sim)
        cost, _ = intensity(sim, 2.3)(record)
        reference = np.sum(np.abs(_transform(sim, [2.3], record)) ** 2)
        self.assertAlmostEqual(cost / reference, 1.0, places=10)

    def test_the_derivative_matches_finite_differences(self):
        sim = _sim()
        record = _record(sim)
        objective = intensity(sim, self.frequencies)
        _, derivative = objective(record)
        rng = np.random.default_rng(2)
        h = 1e-6
        for _ in range(8):
            node = (int(rng.integers(0, sim.N)), int(rng.integers(0, 4)))
            plus, minus = record.copy(), record.copy()
            plus[node] += h
            minus[node] -= h
            reference = (objective(plus)[0] - objective(minus)[0]) / (2.0 * h)
            self.assertAlmostEqual(
                float(derivative[node]) / reference, 1.0, places=6, msg=f"{node}"
            )

    def test_a_weight_selects_one_frequency_and_one_column(self):
        sim = _sim()
        record = _record(sim)
        weights = cp.zeros((len(self.frequencies), 4), dtype=sim.dtype)
        weights[2, 1] = 1.0
        cost, _ = intensity(sim, self.frequencies, weights)(record)
        reference = abs(_transform(sim, self.frequencies, record)[2, 1]) ** 2
        self.assertAlmostEqual(cost / reference, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
