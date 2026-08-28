"""The line search of `Lbfgs`, on costs cheap enough to be checked by hand.

`Lbfgs.search` is the only part of the module whose behaviour is a decision rather
than a formula: it accepts or rejects a trial, and how far it backtracks before it
does is what a driver relies on. The costs below are chosen so that every accepted
`alpha` is a power of `shrink` known in advance:

* `QuadraticTest` is the sanity case, a well-conditioned quadratic the search must
  drive to its minimum with the full step accepted throughout
* `BacktrackTest` puts a wall in the descent direction, so the first two trials
  overshoot into it and the third is the one that decreases the cost
* `BailoutTest` offers a cost that never decreases, which is where the `min_alpha`
  floor is the only thing that ends the search
* `ProjectionTest` pins that the box the driver hands in is respected by the design
  that comes back, not just by the trials inside

`Lbfgs` is array-library agnostic (`@`, `ravel` and `abs().max()` are all numpy
and cupy share), so these run on the CPU and need no GPU.
"""

import unittest

import numpy as np

from cuwave.optimization import Lbfgs


def quadratic(A):
    # 1/2 x^T A x and its gradient, the textbook test problem for a quasi-Newton method
    return lambda x: 0.5 * float(x @ (A @ x)), lambda x: A @ x


class QuadraticTest(unittest.TestCase):
    def test_search_descends_to_the_minimum(self):
        A = np.diag([1.0, 4.0, 16.0])
        cost, grad = quadratic(A)
        x = np.array([1.0, 1.0, 1.0])

        optimizer = Lbfgs(k=10, first_step=0.5)
        history = [cost(x)]
        for _ in range(40):
            x, c, alpha, trials = optimizer.search(x, grad(x), history[-1], cost)
            history.append(c)
            self.assertEqual(trials, 1)  # the curvature estimate needs no backtracking

        self.assertTrue(all(b <= a for a, b in zip(history, history[1:])))
        self.assertLess(history[-1], 1e-12 * history[0])
        self.assertLess(np.abs(x).max(), 1e-6)


class BacktrackTest(unittest.TestCase):
    # 1/2 x^2 with a stiff wall below x = -1, so alpha must backtrack to 1/4
    @staticmethod
    def cost(x):
        return float(0.5 * x[0] ** 2 + 100.0 * max(0.0, -1.0 - x[0]) ** 2)

    def test_backtracks_to_the_first_acceptable_trial(self):
        optimizer = Lbfgs(first_step=10.0, shrink=0.5)
        x = np.array([2.0])

        x, c, alpha, trials = optimizer.search(
            x, np.array([2.0]), self.cost(x), self.cost
        )

        self.assertEqual(trials, 3)
        self.assertAlmostEqual(alpha, 0.25)
        self.assertAlmostEqual(float(x[0]), -0.5)
        self.assertAlmostEqual(c, 0.125)


class BailoutTest(unittest.TestCase):
    def test_gives_up_at_min_alpha(self):
        # a constant cost satisfies no Armijo condition, the margin being negative
        optimizer = Lbfgs(first_step=1.0, shrink=0.5, min_alpha=1e-3)
        x = np.array([1.0])

        _, _, alpha, trials = optimizer.search(x, np.array([1.0]), 1.0, lambda x: 1.0)

        self.assertEqual(trials, 11)  # 2 ** -10 = 9.8e-4 is the first below the floor
        self.assertLessEqual(alpha, 1e-3)


class ProjectionTest(unittest.TestCase):
    def test_the_returned_design_stays_in_the_box(self):
        A = np.diag([1.0, 1.0])
        cost, grad = quadratic(A)
        x = np.array([1.0, 1.0])
        project = lambda x: np.clip(x, 0.5, 1.0)

        optimizer = Lbfgs(first_step=0.4)
        for _ in range(5):
            x, c, _, _ = optimizer.search(x, grad(x), cost(x), cost, project)
            self.assertGreaterEqual(float(x.min()), 0.5)
            self.assertLessEqual(float(x.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
