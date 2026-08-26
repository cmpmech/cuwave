from collections import deque
from collections.abc import Callable

import cupy.typing as cpt


class Adam:
    """Adam optimizer (Kingma & Ba, 2015): https://arxiv.org/abs/1412.6980"""

    def __init__(
        self,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = self.v = 0.0
        self.t = 0

    def step(self, x: cpt.NDArray, grad: cpt.NDArray) -> cpt.NDArray:
        """Update `x` using the Adam rule for one gradient `grad`."""
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * grad**2
        m_hat = self.m / (1.0 - self.b1**self.t)
        v_hat = self.v / (1.0 - self.b2**self.t)
        return x - self.lr * m_hat / (v_hat**0.5 + self.eps)


class Lbfgs:
    """L-BFGS with Armijo line search over the last `k` secant pairs.

    Adapted from https://doi.org/10.33774/coe-2021-qpq2j
    """

    def __init__(
        self,
        k: int = 10,
        lr: float = 1.0,
        first_step: float = 0.05,
        armijo: float = 1e-4,
        shrink: float = 0.5,
        min_alpha: float = 1e-3,
    ) -> None:
        self.lr = lr
        self.pairs = deque(maxlen=k)  # oldest first, drops the oldest when full
        self.x = self.grad = None
        self.first_step = first_step
        self.armijo, self.shrink, self.min_alpha = armijo, shrink, min_alpha

    def search(
        self,
        x: cpt.NDArray,
        grad: cpt.NDArray,
        cost: float,
        f: Callable,
        project: Callable = lambda x: x,
    ) -> tuple[cpt.NDArray, float, float, int]:
        """Armijo backtracking on the proposal `step(x, grad)`.

        Args:
            x: the current design.
            grad: its gradient, and the direction the two-loop recursion turns.
            cost: the objective at `x`, which the sufficient decrease is measured from.
            f: forward only, `f(design) -> cost`, so every trial costs one forward eval.
            project: applied to every trial, a constraint on the design variables.

        Returns:
            (design, cost, alpha, trials) of the accepted trial, which is the last one
            attempted even when `alpha` bottomed out at `min_alpha`.
        """
        step = self.step(x, grad) - x
        if not self.pairs:
            step = step * (self.first_step / float(abs(step).max()))
        slope = float(grad.ravel() @ step.ravel())

        alpha, trials = 1.0, 0
        while True:
            trial = project(x + alpha * step)
            trial_cost = f(trial)
            trials += 1
            if (
                trial_cost <= cost + self.armijo * alpha * slope
                or alpha <= self.min_alpha
            ):
                return trial, trial_cost, alpha, trials
            alpha *= self.shrink

    def step(self, x: cpt.NDArray, grad: cpt.NDArray) -> cpt.NDArray:
        """Optimization step (evaluated in the line `search`)."""
        flat_x, flat_grad = x.ravel(), grad.ravel()
        if self.x is not None:
            self.put(flat_x - self.x, flat_grad - self.grad)
        self.x, self.grad = flat_x, flat_grad
        return x - self.lr * self.iterate(flat_grad).reshape(x.shape)

    def put(self, s: cpt.NDArray, y: cpt.NDArray) -> None:
        """Store secant pair `(s, y)` if it satisfies the curvature condition."""
        if y @ s > 0.0:  # skip pairs violating the curvature condition
            self.pairs.append((s, y))

    def iterate(self, q: cpt.NDArray) -> cpt.NDArray:
        """Two-loop recursion: turn gradient `q` into the L-BFGS descent direction."""
        # backward pass over stored pairs (newest to oldest)
        alpha = []
        for s, y in reversed(self.pairs):
            a = (s @ q) / (y @ s)
            q = q - a * y
            alpha.append(a)

        if self.pairs:
            s, y = self.pairs[-1]
            q = q * ((s @ y) / (y @ y))

        # forward pass building the descent direction (oldest to newest)
        r = q
        for (s, y), a in zip(self.pairs, reversed(alpha)):
            beta = (y @ r) / (y @ s)
            r = r + (a - beta) * s

        return r
