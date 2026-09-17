"""Adjoint contracts of the design maps and penalties, which nothing downstream checks.

Every `Regularization` carries a hand-written `grad`, and a driver chains those adjoints
in the reverse of the order it chained the forward calls. A sign error or a dropped
transpose there still produces a descent direction, so the optimization converges to
something and the mistake never surfaces. The checks below are therefore the same two
identities the module is built on:

* `_spatial_grad` and `_spatial_grad_adjoint` are a transpose pair, which both penalties
  inherit, so a failure there explains every failure after it
* `grad` is the derivative of `__call__`, verified against central differences: directly
  for the scalar penalties, and through the dot-product identity for the maps, whose
  output is an array

The remaining classes pin the properties a driver relies on but no adjoint test would
catch: that the filter preserves a constant, that the projection stays in [0, 1], and
that `set` retunes a live instance the way a `continuation` schedule needs.
"""

import math
import unittest

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.regularization import (
        SIMP,
        DensityFilter,
        Projection,
        Tikhonov,
        TotalVariation,
        _spatial_grad,
        _spatial_grad_adjoint,
        continuation,
    )

STEP = 1e-6


# -------------------------------------- helpers --------------------------------------
def _design(shape=(9, 7), low=0.15, high=0.85, seed=0):
    """A float64 design strictly inside [0, 1], where every map is differentiable."""
    cp.random.seed(seed)
    return cp.random.uniform(low, high, size=shape).astype(cp.float64)


def _directional(f, x, v, h=STEP):
    """Central difference of `f` at `x` along `v`, the directional derivative."""
    return (f(x + h * v) - f(x - h * v)) / (2.0 * h)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class DifferencingTest(unittest.TestCase):
    """`_spatial_grad_adjoint` transposes `_spatial_grad`, under both penalties"""

    def test_the_difference_and_its_adjoint_are_a_transpose_pair(self):
        for shape in [(9, 7), (5, 6, 4)]:
            x, u = _design(shape, seed=1), _design(shape, seed=2)
            for axis in range(len(shape)):
                with self.subTest(shape=shape, axis=axis):
                    forward = float(cp.sum(u * _spatial_grad(x, axis)))
                    backward = float(cp.sum(_spatial_grad_adjoint(u, axis) * x))
                    self.assertAlmostEqual(forward, backward, places=12)

    def test_the_difference_annihilates_a_constant(self):
        x = cp.ones((6, 6), dtype=cp.float64)
        for axis in range(2):
            with self.subTest(axis=axis):
                self.assertAlmostEqual(float(cp.abs(_spatial_grad(x, axis)).max()), 0.0)


# ------------------------ penalties against finite differences -----------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class PenaltyGradientTest(unittest.TestCase):
    """`Tikhonov` and `TotalVariation` are scalar, so `grad` is checked entrywise"""

    def _check(self, penalty, x, places=6):
        gradient = penalty.grad(x)
        for index in [(0, 0), (2, 3), (4, 1), (x.shape[0] - 1, x.shape[1] - 1)]:
            with self.subTest(index=index):
                direction = cp.zeros_like(x)
                direction[index] = 1.0
                expected = _directional(lambda y: float(penalty(y)), x, direction)
                self.assertAlmostEqual(float(gradient[index]), expected, places=places)

    def test_tikhonov_order_zero_matches_central_differences(self):
        self._check(Tikhonov(0.7, order=0), _design())

    def test_tikhonov_order_zero_with_a_reference_matches_central_differences(self):
        x = _design()
        self._check(Tikhonov(0.7, order=0, x_ref=_design(seed=3)), x)

    def test_tikhonov_order_one_matches_central_differences(self):
        self._check(Tikhonov(0.4, order=1), _design())

    def test_tikhonov_order_one_with_a_reference_matches_central_differences(self):
        self._check(Tikhonov(0.4, order=1, x_ref=_design(seed=4)), _design())

    def test_total_variation_matches_central_differences(self):
        self._check(TotalVariation(0.9, eps=1e-2), _design())

    def test_the_penalties_chain_the_outer_derivative(self):
        x = _design()
        for penalty in [Tikhonov(0.7, order=1), TotalVariation(0.9, eps=1e-2)]:
            with self.subTest(penalty=type(penalty).__name__):
                scaled = penalty.grad(x, dy=2.5) - 2.5 * penalty.grad(x)
                self.assertAlmostEqual(float(cp.abs(scaled).max()), 0.0, places=10)

    def test_a_constant_design_costs_nothing_to_smooth(self):
        x = cp.full((6, 6), 0.4, dtype=cp.float64)
        self.assertAlmostEqual(float(Tikhonov(1.0, order=1)(x)), 0.0)


# ------------------------- maps against the adjoint identity -------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class MapAdjointTest(unittest.TestCase):
    """A map returns an array, so `grad` is its transpose against a random `dy`"""

    def _check(self, regularization, x, places=6):
        v, dy = _design(x.shape, seed=5), _design(x.shape, seed=6)
        jacobian = _directional(regularization, x, v)
        forward = float(cp.sum(dy * jacobian))
        backward = float(cp.sum(v * regularization.grad(x, dy)))
        self.assertAlmostEqual(forward, backward, places=places)

    def test_the_density_filter_is_its_own_transpose_against_dy(self):
        x = _design()
        self._check(DensityFilter(2.0, x.shape, dtype=x.dtype), x)

    def test_the_projection_is_its_own_transpose_against_dy(self):
        self._check(Projection(beta=8.0), _design())

    def test_simp_is_its_own_transpose_against_dy(self):
        self._check(SIMP(p=3.0, x_min=1e-3), _design())


# --------------------------- properties a driver relies on ---------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class DensityFilterTest(unittest.TestCase):
    def test_the_filter_preserves_a_uniform_design(self):
        x = cp.full((12, 10), 0.3, dtype=cp.float64)
        filtered = DensityFilter(2.5, x.shape, dtype=x.dtype)(x)
        self.assertAlmostEqual(float(cp.abs(filtered - 0.3).max()), 0.0, places=12)

    def test_the_filter_is_linear_so_its_adjoint_ignores_the_design(self):
        x = _design()
        design_filter = DensityFilter(2.0, x.shape, dtype=x.dtype)
        dy = _design(x.shape, seed=7)
        elsewhere = design_filter.grad(_design(x.shape, seed=8), dy)
        self.assertAlmostEqual(
            float(cp.abs(design_filter.grad(x, dy) - elsewhere).max()), 0.0, places=12
        )

    def test_the_oc_sensitivity_survives_an_empty_design(self):
        x = _design()
        design_filter = DensityFilter(2.0, x.shape, dtype=x.dtype)
        filtered = design_filter.sensitivity(cp.zeros_like(x), _design(x.shape, seed=9))
        self.assertTrue(bool(cp.all(cp.isfinite(filtered))))


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class ProjectionTest(unittest.TestCase):
    def test_a_symmetric_threshold_maps_to_one_half(self):
        x = cp.full((4, 4), 0.5, dtype=cp.float64)
        self.assertAlmostEqual(float(Projection(beta=12.0)(x).mean()), 0.5, places=12)

    def test_an_offset_threshold_maps_to_one_half_only_as_beta_sharpens(self):
        for eta in [0.3, 0.7]:
            with self.subTest(eta=eta):
                x = cp.full((4, 4), eta, dtype=cp.float64)
                gentle = float(Projection(beta=12.0, eta=eta)(x).mean())
                sharp = float(Projection(beta=40.0, eta=eta)(x).mean())
                self.assertLess(abs(sharp - 0.5), abs(gentle - 0.5))
                self.assertAlmostEqual(sharp, 0.5, places=9)

    def test_the_projection_stays_inside_the_unit_interval(self):
        x = cp.linspace(0.0, 1.0, 64, dtype=cp.float64).reshape(8, 8)
        for beta in [1.0, 8.0, 64.0]:
            with self.subTest(beta=beta):
                projected = Projection(beta=beta)(x)
                self.assertGreaterEqual(float(projected.min()), -1e-12)
                self.assertLessEqual(float(projected.max()), 1.0 + 1e-12)

    def test_the_ends_of_the_unit_interval_are_fixed_points(self):
        x = cp.array([[0.0, 1.0]], dtype=cp.float64)
        projected = Projection(beta=5.0)(x)
        self.assertAlmostEqual(float(projected[0, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(projected[0, 1]), 1.0, places=12)

    def test_the_projection_is_monotone(self):
        x = cp.linspace(0.0, 1.0, 32, dtype=cp.float64)
        projected = Projection(beta=10.0)(x)
        self.assertTrue(bool(cp.all(cp.diff(projected) > 0.0)))

    def test_sharpening_beta_steepens_the_map_about_the_threshold(self):
        x = cp.full((4, 4), 0.6, dtype=cp.float64)
        gentle = float(Projection(beta=1.0)(x).mean())
        sharp = float(Projection(beta=20.0)(x).mean())
        self.assertGreater(sharp, gentle)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class SimpTest(unittest.TestCase):
    def test_unit_exponent_without_a_floor_is_the_identity(self):
        x = _design()
        self.assertAlmostEqual(float(cp.abs(SIMP(p=1.0, x_min=0.0)(x) - x).max()), 0.0)

    def test_the_floor_is_the_value_an_empty_design_takes(self):
        x = cp.zeros((4, 4), dtype=cp.float64)
        self.assertAlmostEqual(float(SIMP(p=3.0, x_min=1e-3)(x).max()), 1e-3)

    def test_penalization_pushes_intermediate_densities_down(self):
        x = _design()
        self.assertTrue(bool(cp.all(SIMP(p=3.0, x_min=0.0)(x) < x)))


# ------------------------------- continuation schedules ------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class ContinuationTest(unittest.TestCase):
    def test_set_retunes_a_live_instance(self):
        x = _design()
        projection = Projection(beta=1.0)
        gentle = float(projection(x).std())
        self.assertIs(projection.set(beta=20.0), projection)
        self.assertEqual(projection.beta, 20.0)
        self.assertGreater(float(projection(x).std()), gentle)

    def test_set_refuses_a_parameter_the_regularizer_does_not_have(self):
        with self.assertRaises(AttributeError):
            Projection(beta=1.0).set(gamma=2.0)

    def test_every_scheme_has_one_value_per_iteration(self):
        for scheme in ["constant", "linear", "exponential", "staircase"]:
            with self.subTest(scheme=scheme):
                self.assertEqual(len(continuation(scheme, 17, 1.0, 32.0)), 17)

    def test_the_ramps_start_and_end_where_they_are_told(self):
        for scheme in ["linear", "exponential", "staircase"]:
            with self.subTest(scheme=scheme):
                betas = continuation(scheme, 12, 1.0, 32.0)
                self.assertAlmostEqual(betas[0], 1.0)
                self.assertAlmostEqual(betas[-1], 32.0)

    def test_the_constant_scheme_is_the_reference_that_never_sharpens(self):
        self.assertEqual(continuation("constant", 5, 3.0, 32.0), [3.0] * 5)

    def test_the_ramps_never_decrease(self):
        for scheme in ["linear", "exponential", "staircase"]:
            with self.subTest(scheme=scheme):
                betas = continuation(scheme, 20, 1.0, 64.0)
                self.assertTrue(all(b >= a for a, b in zip(betas, betas[1:])))

    def test_the_exponential_ramp_spends_its_early_iterations_near_the_start(self):
        linear = continuation("linear", 21, 1.0, 1024.0)
        exponential = continuation("exponential", 21, 1.0, 1024.0)
        self.assertLess(exponential[10], linear[10])
        self.assertAlmostEqual(exponential[10], math.sqrt(1024.0), places=6)

    def test_the_staircase_holds_each_level_for_a_block_of_iterations(self):
        betas = continuation("staircase", 16, 1.0, 8.0, stages=4)
        self.assertEqual(len(set(betas)), 4)
        self.assertEqual(betas[:4], [1.0] * 4)

    def test_an_unknown_scheme_is_refused(self):
        with self.assertRaises(ValueError):
            continuation("cosine", 10, 1.0, 8.0)


if __name__ == "__main__":
    unittest.main()
