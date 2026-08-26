from collections import deque


class Adam:
    """Adam optimizer (Kingma & Ba, 2015): https://arxiv.org/abs/1412.6980"""

    def __init__(self, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = self.v = 0.0
        self.t = 0

    def step(self, x, grad):
        """Update `x` using the Adam rule for one gradient `grad`."""
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * grad**2
        m_hat = self.m / (1.0 - self.b1**self.t)
        v_hat = self.v / (1.0 - self.b2**self.t)
        return x - self.lr * m_hat / (v_hat**0.5 + self.eps)


class Lbfgs:
    """L-BFGS with Armijo line search, via two-loop recursion over the last ``k`` secant pairs: adapted from https://doi.org/10.33774/coe-2021-qpq2j"""

    def __init__(
        self, k=10, lr=1.0, first_step=0.05, armijo=1e-4, shrink=0.5, min_alpha=1e-3
    ):
        self.lr = lr
        self.pairs = deque(maxlen=k)  # oldest first, drops the oldest when full
        self.x = self.grad = None
        self.first_step = first_step
        self.armijo, self.shrink, self.min_alpha = armijo, shrink, min_alpha

    def search(self, x, grad, cost, f, project=lambda x: x):
        """Armijo backtracking on the proposal: (design, cost, alpha, trials).
        `f(design) -> cost` is forward only, so a trial costs a forward eval.
        `project` is applied to every trial: a constraint on the design variables `x`.
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

    def step(self, x, grad):
        """Optimization step (evaluated in the line `search`)."""
        flat_x, flat_grad = x.ravel(), grad.ravel()
        if self.x is not None:
            self.put(flat_x - self.x, flat_grad - self.grad)
        self.x, self.grad = flat_x, flat_grad
        return x - self.lr * self.iterate(flat_grad).reshape(x.shape)

    def put(self, s, y):
        """Store secant pair `(s, y)` if it satisfies the curvature condition."""
        if y @ s > 0.0:  # skip pairs violating the curvature condition
            self.pairs.append((s, y))

    def iterate(self, q):
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
