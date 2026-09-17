"""Scoring contracts of the damage metrics, whose orientation is the thing that inverts.

`evals` scores a recovered indicator against a truth, and the whole module rests on one
convention the arithmetic never states twice: **damage is a low indicator value**. Both
labels and predictions are formed as `x < threshold`, `_ranked` sorts ascending to get a
descending damage score, and `roc_auc` negates before searching. Flip one of them and
every metric still returns a number in [0, 1], plausible and wrong, so the orientation
is pinned first and separately.

The rest are the cases a driver hits at the edges of a run: a perfect reconstruction, a
fully inverted one, an all-grey field whose scores are pure ties, and the degenerate
inputs that have no defined answer. Those last return `NAN` rather than raising, because
an inversion scores itself every iteration and must not die on the iteration where the
recovered field happens to be uniform.
"""

import math
import unittest

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.evals import (
        confusion,
        f1_score,
        false_positive_rate,
        l2_error,
        non_discreteness,
        pr_auc,
        precision,
        recall,
        roc_auc,
    )

# two damaged nodes at 0 and three intact at 1, so the midpoint threshold is 0.5
TRUTH = [0.0, 0.0, 1.0, 1.0, 1.0]


# -------------------------------------- helpers --------------------------------------
def _field(values):
    """A float64 indicator field on the device, damage low and intact high."""
    return cp.asarray(values, dtype=cp.float64)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class OrientationTest(unittest.TestCase):
    """Damage is the low value, the one convention every metric in the module shares"""

    def test_a_perfect_reconstruction_has_no_false_counts(self):
        truth = _field(TRUTH)
        self.assertEqual(confusion(truth, truth), (2, 0, 0, 3))

    def test_an_inverted_reconstruction_finds_no_damage_at_all(self):
        truth = _field(TRUTH)
        self.assertEqual(confusion(1.0 - truth, truth), (0, 3, 2, 0))

    def test_inverting_the_ranking_sends_the_roc_to_zero(self):
        truth = _field(TRUTH)
        self.assertAlmostEqual(roc_auc(truth, truth), 1.0)
        self.assertAlmostEqual(roc_auc(1.0 - truth, truth), 0.0)

    def test_the_counts_partition_the_grid(self):
        truth = _field(TRUTH)
        for values in [TRUTH, [1.0] * 5, [0.0] * 5, [0.0, 1.0, 0.0, 1.0, 0.0]]:
            with self.subTest(values=values):
                self.assertEqual(sum(confusion(_field(values), truth)), len(TRUTH))

    def test_the_metrics_ignore_the_grid_shape(self):
        field = _field([[0.0, 0.0], [1.0, 1.0]])
        truth = _field([[0.0, 1.0], [1.0, 1.0]])
        flat = confusion(field.ravel(), truth.ravel())
        self.assertEqual(confusion(field, truth), flat)

    def test_a_shape_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            confusion(cp.ones(3), cp.ones(4))


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class RateTest(unittest.TestCase):
    """The four counting metrics against hand-checked confusion matrices"""

    def test_one_false_positive_costs_precision_but_not_recall(self):
        field, truth = _field([0.0, 0.0, 0.0, 1.0, 1.0]), _field(TRUTH)
        self.assertEqual(confusion(field, truth), (2, 1, 0, 2))
        self.assertAlmostEqual(precision(field, truth), 2 / 3)
        self.assertAlmostEqual(recall(field, truth), 1.0)
        self.assertAlmostEqual(false_positive_rate(field, truth), 1 / 3)
        self.assertAlmostEqual(f1_score(field, truth), 0.8)

    def test_one_miss_costs_recall_but_not_precision(self):
        field, truth = _field([0.0, 1.0, 1.0, 1.0, 1.0]), _field(TRUTH)
        self.assertEqual(confusion(field, truth), (1, 0, 1, 3))
        self.assertAlmostEqual(precision(field, truth), 1.0)
        self.assertAlmostEqual(recall(field, truth), 0.5)
        self.assertAlmostEqual(f1_score(field, truth), 2 / 3)

    def test_the_f1_is_the_harmonic_mean_of_the_other_two(self):
        field, truth = _field([0.0, 0.0, 0.0, 1.0, 1.0]), _field(TRUTH)
        p, r = precision(field, truth), recall(field, truth)
        self.assertAlmostEqual(f1_score(field, truth), 2 * p * r / (p + r))

    def test_an_explicit_threshold_overrides_the_midpoint(self):
        field, truth = _field([0.2, 0.4, 0.6, 0.8, 0.9]), _field(TRUTH)
        self.assertEqual(confusion(field, truth, threshold=0.3), (1, 0, 1, 3))
        self.assertEqual(confusion(field, truth, threshold=0.7), (2, 1, 0, 2))


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class AreaTest(unittest.TestCase):
    """`pr_auc` and `roc_auc` rank rather than threshold, so ties are the real case"""

    def test_a_separable_ranking_scores_one(self):
        truth = _field(TRUTH)
        self.assertAlmostEqual(pr_auc(truth, truth), 1.0)
        self.assertAlmostEqual(roc_auc(truth, truth), 1.0)

    def test_a_ranking_that_is_all_ties_is_the_coin_flip(self):
        truth = _field(TRUTH)
        self.assertAlmostEqual(roc_auc(_field([0.5] * 5), truth), 0.5)

    def test_a_ranking_that_is_all_ties_scores_the_base_rate_on_the_pr_curve(self):
        truth = _field(TRUTH)
        self.assertAlmostEqual(pr_auc(_field([0.5] * 5), truth), 2 / 5)

    def test_a_graded_field_still_ranks_even_where_a_threshold_would_not(self):
        truth = _field(TRUTH)
        graded = _field([0.30, 0.35, 0.40, 0.45, 0.50])
        self.assertEqual(confusion(graded, truth), (2, 2, 0, 1))
        self.assertAlmostEqual(roc_auc(graded, truth), 1.0)

    def test_the_areas_are_invariant_under_a_monotone_rescaling(self):
        truth = _field(TRUTH)
        field = _field([0.1, 0.2, 0.7, 0.8, 0.9])
        for area in [pr_auc, roc_auc]:
            with self.subTest(area=area.__name__):
                self.assertAlmostEqual(
                    area(field, truth), area(0.5 * field + 0.25, truth)
                )


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class DegenerateTest(unittest.TestCase):
    """An inversion scores itself every iteration, so an undefined metric returns NAN"""

    def test_a_truth_without_damage_leaves_the_rates_undefined(self):
        intact = _field([1.0] * 4)
        for metric in [recall, roc_auc, pr_auc]:
            with self.subTest(metric=metric.__name__):
                self.assertTrue(math.isnan(metric(intact, intact)))

    def test_a_field_flagging_nothing_leaves_the_precision_undefined(self):
        self.assertTrue(math.isnan(precision(_field([1.0] * 5), _field(TRUTH))))

    def test_the_f1_scores_zero_rather_than_nan_when_a_miss_is_the_reason(self):
        self.assertAlmostEqual(f1_score(_field([1.0] * 5), _field(TRUTH)), 0.0)

    def test_an_empty_region_leaves_the_greyness_undefined(self):
        region = cp.zeros(4, dtype=cp.bool_)
        self.assertTrue(math.isnan(non_discreteness(_field([1.0] * 4), region)))


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class GreynessTest(unittest.TestCase):
    """`non_discreteness` takes no truth, so it scores a topology optimization too"""

    def test_a_discrete_design_is_not_grey_at_all(self):
        self.assertAlmostEqual(non_discreteness(_field([0.0, 1.0, 0.0, 1.0])), 0.0)

    def test_a_uniformly_grey_design_saturates_the_measure(self):
        self.assertAlmostEqual(non_discreteness(_field([0.5] * 4)), 1.0)

    def test_the_measure_is_symmetric_about_one_half(self):
        low, high = _field([0.3] * 4), _field([0.7] * 4)
        self.assertAlmostEqual(non_discreteness(low), non_discreteness(high))

    def test_the_region_selects_what_is_scored(self):
        field = _field([0.5, 0.5, 0.0, 1.0])
        region = cp.asarray([True, True, False, False])
        self.assertAlmostEqual(non_discreteness(field, region), 1.0)
        self.assertAlmostEqual(non_discreteness(field), 0.5)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class L2ErrorTest(unittest.TestCase):
    def test_an_exact_reconstruction_has_no_error(self):
        truth = _field(TRUTH)
        self.assertAlmostEqual(l2_error(truth, truth), 0.0)

    def test_the_absolute_error_is_the_norm_of_the_difference(self):
        truth = _field(TRUTH)
        expected = float(cp.linalg.norm(truth))
        self.assertAlmostEqual(l2_error(cp.zeros(5), truth, relative=False), expected)

    def test_the_relative_error_is_normalized_by_the_truth(self):
        truth = _field(TRUTH)
        absolute = l2_error(cp.zeros(5), truth, relative=False)
        relative = l2_error(cp.zeros(5), truth)
        self.assertAlmostEqual(relative, absolute / float(cp.linalg.norm(truth)))
        self.assertAlmostEqual(relative, 1.0)

    def test_the_error_ignores_the_grid_shape(self):
        field = _field([[0.0, 0.2], [0.9, 1.0]])
        truth = _field([[0.0, 0.0], [1.0, 1.0]])
        self.assertAlmostEqual(
            l2_error(field, truth), l2_error(field.ravel(), truth.ravel())
        )


if __name__ == "__main__":
    unittest.main()
