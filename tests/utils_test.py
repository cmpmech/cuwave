"""Contracts of the spectral objective and the time-dispersion transforms.

`intensity` is a linear transform of the sensor record followed by a squared magnitude,
and both halves are easy to get subtly wrong: the sign and the `dt` of the transform
decide whether the reported spectrum is the physical one, and its adjoint is one
conjugation away from an expression that agrees on real data and disagrees on
everything else. So the transform is pinned against `numpy.fft` and its derivative
against central differences.

`tdt` / `itdt` are only right with the time origin of each record right, a source row
`t` sitting at `t dt` and a sensor row at `(t + 1) dt`, which no round trip can see. So
they are pinned on a real leapfrog run against the exact 1D solution.

`devices` spreads the shots over threads, one stream each, so it must reproduce the
serial cost and gradient; two entries naming the same GPU exercise that on one card.
"""

import unittest
from dataclasses import replace

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.scalar import ScalarWave
    from cuwave.signals import ricker
    from cuwave.sensitivity import reconstruction_sensitivity, sensitivity
    from cuwave.utils import (
        Sensors,
        intensity,
        itdt,
        leapfrog_frequency,
        measure,
        misfit,
        misfit_gradient,
        point_source,
        shots,
        tdt,
    )
    from cuwave.wave import simulate, stable_dt


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


def _line(space_order=8, points=6, frequency=1.0):
    """1D unit medium at `points` nodes per minimum wavelength (2.5 `frequency`)."""
    dx = 1.0 / (2.5 * frequency * points)
    dt = 0.99 * stable_dt((dx,), 1.0, space_order, "taylor")
    Nx = (int(60.0 / dx) + 3,)
    return ScalarWave(
        Nx,
        (dx,),
        int(25.0 / dt),
        dt,
        (128,),
        precision="float64",
        space_order=space_order,
        stencil="taylor",  # resolved enough that Taylor beats the fitted row
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


# ---------------------------------- time dispersion ----------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class TimeDispersionTest(unittest.TestCase):
    frequency = 1.0

    def _trace(self, sim):
        return ricker(
            np.arange(sim.N) * sim.dt, 1.0, self.frequency, 1.5 / self.frequency
        )

    def _shot(self, sim, signal):
        """The record 20 length units from the source, and the exact 1D answer there."""
        x_source = 0.5 * (sim.Nx[0] - 3) * sim.dx[0]
        node = int(round((x_source + 20.0) / sim.dx[0])) + 1
        sensors = cp.asarray([[node]], dtype=cp.int32)
        source = point_source(sim, [[x_source]], signal)
        _, um = simulate(sim, source, cp.ones(sim.Nx_padded), sensors=sensors)
        # u = 1/2 int_0^(t - r) f: the Ricker integrates to lag * exp(-a lag^2)
        a, delay = (np.pi * self.frequency) ** 2, 1.5 / self.frequency
        lag = np.arange(1, sim.N + 1) * sim.dt - ((node - 1) * sim.dx[0] - x_source)
        lag = np.maximum(lag, 0.0) - delay
        exact = 0.5 * (lag * np.exp(-a * lag**2) + delay * np.exp(-a * delay**2))
        return um.get()[:, 0], exact

    def test_the_transforms_remove_the_leapfrog_time_error(self):
        sim = _line()
        plain, exact = self._shot(sim, self._trace(sim))
        record, _ = self._shot(sim, tdt(sim, self._trace(sim)))
        corrected = itdt(sim, record)
        error = np.linalg.norm(corrected - exact) / np.linalg.norm(exact)
        self.assertLess(error, 2e-3)
        self.assertLess(
            error, 0.01 * np.linalg.norm(plain - exact) / np.linalg.norm(exact)
        )

    def test_the_inverse_undoes_the_transform(self):
        sim = _line()
        signal = self._trace(sim)
        for offset in (0, 1):
            back = itdt(sim, tdt(sim, signal, offset), offset)
            self.assertLess(np.abs(back - signal).max(), 1e-6 * np.abs(signal).max())

    def test_a_device_record_keeps_its_module_shape_and_dtype(self):
        sim = _line()
        record = cp.ones((sim.N, 3), dtype=cp.float32)
        out = itdt(sim, record)
        self.assertIsInstance(out, cp.ndarray)
        self.assertEqual((out.shape, out.dtype), (record.shape, record.dtype))

    def test_order_2_warns(self):
        sim = _line(space_order=2)
        with self.assertWarns(RuntimeWarning):
            tdt(sim, self._trace(sim))

    def test_the_leapfrog_frequency_maps_back_onto_the_requested_one(self):
        sim = _line()
        f = leapfrog_frequency(sim, [0.5, 3.0])
        warped = np.sin(np.pi * f * sim.dt) / (np.pi * sim.dt)
        np.testing.assert_allclose(warped, [0.5, 3.0], rtol=1e-12)
        with self.assertRaises(ValueError):
            leapfrog_frequency(sim, 1.01 / (np.pi * sim.dt))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class DevicesTest(unittest.TestCase):
    def setUp(self):
        sim = _sim(N=80)
        damping = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
        damping[:, 24:] = 2.0
        self.sim = replace(sim, damping=damping)
        coords = [[0.5, 0.5], [1.5, 0.5], [2.5, 1.0]]
        self.sources = shots(
            self.sim, coords, ricker(np.arange(80) * self.sim.dt, 1.0, 4.0)
        )
        self.sensors = Sensors(self.sim, [[0.5, 2.5], [1.5, 2.5], [2.5, 2.5]])
        self.gamma = cp.ones(self.sim.Nx_padded, dtype=self.sim.dtype)
        self.gamma[12:18, 12:18] = 0.8
        self.observed = measure(
            self.sim, self.sources, cp.ones_like(self.gamma), self.sensors
        )

    def test_the_spread_traces_match_the_serial_ones(self):
        serial = measure(self.sim, self.sources, self.gamma, self.sensors)
        spread = measure(self.sim, self.sources, self.gamma, self.sensors, [0, 0])
        for a, b in zip(serial, spread):
            np.testing.assert_array_equal(a.get(), b.get())

    def test_the_spread_cost_and_gradient_match_the_serial_ones(self):
        args = (self.sim, self.sources, self.gamma, self.sensors, self.observed)
        for adjoint in (sensitivity, reconstruction_sensitivity):
            cost, gradient = misfit_gradient(*args, adjoint=adjoint)
            spread_cost, spread = misfit_gradient(
                *args, adjoint=adjoint, devices=[0, 0]
            )
            self.assertAlmostEqual(spread_cost / cost, 1.0, places=12)
            np.testing.assert_allclose(
                spread.get(), gradient.get(), rtol=1e-10, atol=1e-14
            )
            self.assertAlmostEqual(misfit(*args, devices=[0, 0]) / cost, 1.0, places=12)
