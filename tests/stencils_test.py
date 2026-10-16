"""Contracts of the coefficient tables the kernels compile against.

An optimized row trades formal order for band: it must stay consistent, keep the phase
velocity error under `DISPERSION_TOL` over a band wider than the Taylor row's, and leave
order 2 and the graded wall rows Taylor. `stable_dt` must follow the row, since a fitted
symbol peaks higher on the Nyquist mode. And the compiled step must apply the fitted
symbol, the scalar flux and the staggered elastic difference alike, which is checked on a
plane wave rather than trusted from the preamble string.
"""

import unittest

import numpy as np

from cuwave.stencils import (
    DISPERSION_TOL,
    cell_coefficients,
    preamble,
    staggered_weights,
    weights,
)

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # cupy missing or no GPU
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.elastic import ElasticWave
    from cuwave.scalar import ScalarWave
    from cuwave.wave import compile_kernels, stable_dt


# -------------------------------------- helpers --------------------------------------
def _symbol(c, theta, staggered):
    """Each row's symbol over its exact one: 1 for an exact derivative."""
    k, theta = np.arange(1, len(c) + 1), np.asarray(theta)
    ratio = 2.0 * np.sin(np.outer(theta, k - 0.5)) @ c / theta
    return ratio if staggered else ratio * 2.0 * np.sin(theta / 2.0) / theta


def _band(c, staggered):
    """The widest band from 0 whose phase velocity error stays under `DISPERSION_TOL`."""
    theta = np.linspace(1e-3, np.pi, 20000)
    ratio = _symbol(c, theta, staggered)
    error = np.abs(ratio - 1.0 if staggered else np.sqrt(ratio) - 1.0)
    bad = error > DISPERSION_TOL * (1.0 + 1e-6)
    return theta[np.argmax(bad)] if bad.any() else np.pi


def _plane_wave_ratio(sim, theta):
    """Least-squares `(u2 - 2 u1) / u1` of one step on sin along the last axis, deep inside."""
    step = sim.define_step(
        compile_kernels(sim), sim.build_materials(cp.ones(sim.Nx_padded, sim.dtype))
    )
    u1 = cp.zeros(sim.field_shape, dtype=sim.dtype)
    x = cp.arange(sim.Nx_padded[-1], dtype=sim.dtype)
    u1[...] = cp.sin(theta * x)
    u2 = cp.zeros_like(u1)
    step(cp.zeros_like(u1), u1, u2)
    deep = (..., slice(2 * sim.space_order, sim.Nx[-1] - 2 * sim.space_order))
    a, b = u2[deep] - 2.0 * u1[deep], u1[deep]
    return float(cp.sum(a * b) / cp.sum(b * b))


# ---------------------------------- the fitted rows ----------------------------------
class OptimizedRowTest(unittest.TestCase):
    def test_the_rows_stay_consistent(self):
        for R in (2, 3, 4, 6):
            for fit in (cell_coefficients, staggered_weights):
                with self.subTest(R=R, fit=fit.__name__):
                    c = fit(R, "optimized")
                    self.assertAlmostEqual(float(c @ (2 * np.arange(1, R + 1) - 1)), 1.0)

    def test_the_band_is_wider_than_taylors_under_the_tolerance(self):
        for R in (2, 3, 4, 6):
            for fit, staggered in ((cell_coefficients, False), (staggered_weights, True)):
                with self.subTest(R=R, staggered=staggered):
                    taylor = _band(fit(R), staggered)
                    optimized = _band(fit(R, "optimized"), staggered)
                    self.assertGreater(optimized, 1.2 * taylor)

    def test_order_2_and_the_wall_rows_stay_taylor(self):
        np.testing.assert_allclose(cell_coefficients(1, "optimized"), [1.0])
        np.testing.assert_allclose(staggered_weights(1, "optimized"), [1.0])
        taylor, optimized = preamble(8).split("\n"), preamble(8, "optimized").split("\n")
        for line, other in zip(taylor[1:3], optimized[1:3]):
            rows, fitted = line.split("}, {"), other.split("}, {")
            self.assertEqual(rows[:-1], fitted[:-1])
            self.assertNotEqual(rows[-1], fitted[-1])

    def test_an_unknown_stencil_is_refused(self):
        with self.assertRaises(ValueError):
            cell_coefficients(3, "spectral")


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class CompiledRowTest(unittest.TestCase):
    def test_stable_dt_follows_the_row(self):
        for R in (1, 2, 4):
            with self.subTest(R=R):
                taylor = stable_dt((1.0,), 1.0, 2 * R, "taylor")
                self.assertAlmostEqual(taylor, 2.0 / np.sqrt(np.abs(weights(R)).sum()))
                optimized = stable_dt((1.0,), 1.0, 2 * R, "optimized")
                self.assertLessEqual(optimized, taylor)
                nyquist = _symbol(cell_coefficients(R, "optimized"), [np.pi], False)[0]
                self.assertAlmostEqual(optimized, 2.0 / (np.pi * np.sqrt(nyquist)))

    def test_the_step_applies_the_fitted_symbol(self):
        theta = 1.4
        for order in (4, 8):
            for cls, staggered in ((ScalarWave, False), (ElasticWave, True)):
                with self.subTest(order=order, equation=cls.__name__):
                    kwargs = (
                        {"wavespeed": 1.0, "density": 1.0}
                        if cls is ScalarWave
                        else {"density": 1.0, "wavespeed_p": 1.0, "wavespeed_s": 0.5}
                    )
                    ratios = []
                    for stencil in ("taylor", "optimized"):
                        sim = cls(
                            (200,),
                            (1.0,),
                            1,
                            0.3,
                            (128,),
                            precision="float64",
                            space_order=order,
                            stencil=stencil,
                            **kwargs,
                        )
                        ratios.append(_plane_wave_ratio(sim, theta))
                    R = order // 2
                    fit = staggered_weights if staggered else cell_coefficients
                    symbols = [
                        _symbol(fit(R, s), [theta], staggered)[0] ** (1 + staggered)
                        for s in ("taylor", "optimized")
                    ]
                    self.assertAlmostEqual(
                        ratios[1] / ratios[0], symbols[1] / symbols[0], places=10
                    )


if __name__ == "__main__":
    unittest.main()
