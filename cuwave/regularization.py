"""Regularization and design-map penalties for gradient-based optimization.

A `Regularization` is both a differentiable map (`__call__`) and its own adjoint
(`grad`), so filters, projections, and penalties compose the same way the simulation's forward/adjoint pair does: chain the calls forward, chain `grad` backward. `set` lets a `continuation` schedule mutate a live instance's parameters between iterations without rebuilding the pipeline.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence

import cupy as cp
import cupy.typing as cpt
import cupyx.scipy.ndimage as ndi
import numpy.typing as npt


# -------------------------------------- helpers --------------------------------------
def _axis_slice(ndim: int, axis: int, cut: int | slice) -> tuple:
    """Build an ndim-length index tuple selecting `cut` along `axis`, `slice(None)` elsewhere."""
    index = [slice(None)] * ndim
    index[axis] = cut
    return tuple(index)


def _spatial_grad(x: cpt.NDArray, axis: int) -> cpt.NDArray:
    """forward difference along `axis`, zero-padded at the far boundary"""
    head = _axis_slice(x.ndim, axis, slice(None, -1))
    tail = _axis_slice(x.ndim, axis, slice(1, None))
    d = cp.zeros_like(x)
    d[head] = x[tail] - x[head]
    return d


def _spatial_grad_adjoint(u: cpt.NDArray, axis: int) -> cpt.NDArray:
    """transpose of `_spatial_grad`: the negative divergence of `u`"""
    head = _axis_slice(u.ndim, axis, slice(None, -1))
    tail = _axis_slice(u.ndim, axis, slice(1, None))
    inner = u[head]
    out = cp.zeros_like(u)
    out[head] -= inner
    out[tail] += inner
    return out


# ------------------------------- regularization classes ------------------------------
class Regularization(abc.ABC):
    """Forward pass in `__call__(x)`, backward pass in `grad(x, dy)`, re-tuned by `set`."""

    @abc.abstractmethod
    def __call__(self, x: cpt.NDArray) -> cpt.NDArray:
        """Apply the map, or evaluate the penalty."""

    @abc.abstractmethod
    def grad(self, x: cpt.NDArray, dy: cpt.NDArray | float = 1.0) -> cpt.NDArray:
        """Gradient with respect to `x`, given the gradient `dy` towards the output."""

    def set(self, **params) -> Regularization:
        """Update settings of the regularizer: useful in continuation schemes."""
        for name, value in params.items():
            if not hasattr(self, name):
                raise AttributeError(f"{type(self).__name__} has no parameter {name!r}")
            setattr(self, name, value)
        return self


# ------------------------------ design-map modification ------------------------------
class DensityFilter(Regularization):
    """Conic density filter on a structured grid of any dimension, with an OC variant.

    Args:
        rmin: filter radius in nodes, which sets the conic kernel's support.
        shape: the design field's shape, which fixes the dimension of the kernel and
            the normalization `Hs` it is built for.
        dtype: kernel dtype, matched to the design field to keep the convolution
            in one precision.
    """

    def __init__(
        self, rmin: float, shape: tuple[int, ...], dtype: npt.DTypeLike | None = None
    ) -> None:
        ceil_r = int(math.ceil(rmin))
        taps = cp.arange(-ceil_r, ceil_r + 1)
        offsets = cp.meshgrid(*(taps,) * len(shape), indexing="ij")
        self.kernel = cp.maximum(0.0, rmin - cp.sqrt(sum(k**2 for k in offsets)))
        if dtype is not None:
            self.kernel = self.kernel.astype(dtype)
        self.Hs = ndi.convolve(
            cp.ones(shape, self.kernel.dtype), self.kernel, mode="constant", cval=0.0
        )

    def __call__(self, x: cpt.NDArray) -> cpt.NDArray:
        """Filtered design: convolve `x` with the conic kernel, normalized by `Hs`."""
        return ndi.convolve(x, self.kernel, mode="constant", cval=0.0) / self.Hs

    def grad(self, x: cpt.NDArray, dy: cpt.NDArray | float = 1.0) -> cpt.NDArray:
        """Adjoint of the filter applied to `dy`; `x` is unused since the filter is linear."""
        return ndi.convolve(dy / self.Hs, self.kernel, mode="constant", cval=0.0)

    def sensitivity(self, rho: cpt.NDArray, dc: cpt.NDArray) -> cpt.NDArray:
        """Filter sensitivities `dc` at density `rho`, the OC scheme in place of `grad`."""
        num = ndi.convolve(rho * dc, self.kernel, mode="constant", cval=0.0)
        return num / (cp.maximum(rho, 1e-3) * self.Hs)


class Projection(Regularization):
    """Smoothed Heaviside about threshold `eta`, with sharpness `beta`."""

    def __init__(self, beta: float, eta: float = 0.5) -> None:
        self.beta, self.eta = beta, eta

    def _ends(self) -> tuple[float, float]:
        """tanh at the two ends of [0, 1], used to rescale `__call__` and `grad`."""
        return math.tanh(self.beta * self.eta), math.tanh(self.beta * (1.0 - self.eta))

    def __call__(self, x: cpt.NDArray) -> cpt.NDArray:
        """Projected design: rescaled `tanh(beta * (x - eta))`, mapped onto [0, 1]."""
        a, b = self._ends()
        return (a + cp.tanh(self.beta * (x - self.eta))) / (a + b)

    def grad(self, x: cpt.NDArray, dy: cpt.NDArray | float = 1.0) -> cpt.NDArray:
        """Adjoint of the projection: `dy` scaled by the local tanh derivative."""
        a, b = self._ends()
        t = cp.tanh(self.beta * (x - self.eta))
        return dy * self.beta * (1.0 - t * t) / (a + b)


class SIMP(Regularization):
    """Power-law penalization making intermediate designs uneconomical (Bendsoe 1989).

    See https://doi.org/10.1007/BF01650949
    """

    def __init__(self, p: float = 3.0, x_min: float = 0.0) -> None:
        self.p, self.x_min = p, x_min

    def __call__(self, x: cpt.NDArray) -> cpt.NDArray:
        """Penalized design: `x_min + (1 - x_min) * x**p`."""
        # x in [0, 1] is assumed, as a fractional p needs x >= 0
        return self.x_min + (1.0 - self.x_min) * x**self.p

    def grad(self, x: cpt.NDArray, dy: cpt.NDArray | float = 1.0) -> cpt.NDArray:
        """Adjoint of the penalization: `dy` scaled by the power-law derivative."""
        return dy * (1.0 - self.x_min) * self.p * x ** (self.p - 1.0)


# ----------------------------- penalization in objective -----------------------------
class Tikhonov(Regularization):
    """L2 penalty, damping toward a reference or smoothing by penalizing the gradient.

    Args:
        alpha: penalty weight.
        order: 0 penalizes deviation from `x_ref`, 1 penalizes the spatial gradient.
        x_ref: the reference `order` 0 damps toward, or None for zero.
    """

    def __init__(
        self, alpha: float, order: int = 0, x_ref: cpt.NDArray | None = None
    ) -> None:
        if order not in (0, 1):
            raise ValueError("order must be 0 (damping) or 1 (smoothing)")
        self.alpha, self.order, self.x_ref = alpha, order, x_ref

    def _residual(self, x: cpt.NDArray) -> cpt.NDArray:
        """`x` relative to `x_ref`, or `x` itself when no reference is set."""
        return x if self.x_ref is None else x - self.x_ref

    def __call__(self, x: cpt.NDArray) -> cpt.NDArray:
        """Penalty value: `0.5 * alpha` times the squared residual, or its squared gradient."""
        r = self._residual(x)
        if self.order == 0:
            return 0.5 * self.alpha * cp.sum(r * r)
        squares = (cp.sum(_spatial_grad(r, axis) ** 2) for axis in range(r.ndim))
        return 0.5 * self.alpha * sum(squares)

    def grad(self, x: cpt.NDArray, dy: cpt.NDArray | float = 1.0) -> cpt.NDArray:
        """Adjoint of the penalty: `dy * alpha` times the residual, or its Laplacian."""
        r = self._residual(x)
        if self.order == 0:
            return (dy * self.alpha) * r
        out = cp.zeros_like(r)
        for axis in range(r.ndim):
            out += _spatial_grad_adjoint(_spatial_grad(r, axis), axis)
        return (dy * self.alpha) * out


class TotalVariation(Regularization):
    """Smoothed isotropic total variation, `alpha * sum sqrt(|D x|**2 + eps**2)`"""

    def __init__(self, alpha: float, eps: float = 1e-3) -> None:
        self.alpha, self.eps = alpha, eps

    def _magnitude(self, diffs: Sequence[cpt.NDArray]) -> cpt.NDArray:
        """Smoothed gradient magnitude `sqrt(sum(d**2) + eps**2)` over the differences `diffs`."""
        total = self.eps**2
        for d in diffs:
            total = total + d * d
        return cp.sqrt(total)

    def __call__(self, x: cpt.NDArray) -> cpt.NDArray:
        """Penalty value: `alpha` times the summed smoothed gradient magnitude of `x`."""
        diffs = [_spatial_grad(x, axis) for axis in range(x.ndim)]
        return self.alpha * cp.sum(self._magnitude(diffs))

    def grad(self, x: cpt.NDArray, dy: cpt.NDArray | float = 1.0) -> cpt.NDArray:
        """Adjoint of the penalty: divergence of the normalized gradient, scaled by `dy * alpha`."""
        diffs = [_spatial_grad(x, axis) for axis in range(x.ndim)]
        magnitude = self._magnitude(diffs)
        out = cp.zeros_like(x)
        for axis, d in enumerate(diffs):
            out += _spatial_grad_adjoint(d / magnitude, axis)
        return (dy * self.alpha) * out


# ------------------------------ continuation schedules -------------------------------
def continuation(
    scheme: str, iters: int, start: float, stop: float, stages: int = 4
) -> list[float]:
    """`iters` values ramping `start` -> `stop`, a `Projection` sharpness schedule

    * `"constant"`: `start` throughout, the reference
    * `"linear"`: equal increments, so most of the run is already sharp
    * `"exponential"`: equal factors, spending the early iterations near `start`
    * `"staircase"`: `stages` levels held for `iters // stages` iterations
      each, the classic continuation (Wang, Lazarov & Sigmund 2011,
      https://doi.org/10.1007/s00158-010-0602-y)
    """
    last = max(iters - 1, 1)
    if scheme == "constant":
        return [start] * iters
    if scheme == "linear":
        return [start + (stop - start) * i / last for i in range(iters)]
    if scheme == "exponential":
        return [start * (stop / start) ** (i / last) for i in range(iters)]
    if scheme == "staircase":  # the exponential ramp, held over `stages` levels
        every, levels = max(iters // stages, 1), max(stages - 1, 1)
        factors = (min(i // every, levels) / levels for i in range(iters))
        return [start * (stop / start) ** f for f in factors]
    raise ValueError(f"unknown continuation scheme {scheme!r}")
