"""Regularization and design-map penalties for gradient-based optimization.

A `Regularization` is both a differentiable map (`__call__`) and its own adjoint
(`grad`), so filters, projections, and penalties compose the same way the simulation's
forward/adjoint pair does: chain the calls forward, chain `grad` backward. `set` lets a
`continuation` schedule mutate a live instance's parameters between iterations without
rebuilding the pipeline.
"""

from __future__ import annotations

import abc
import math
from pathlib import Path
from collections.abc import Sequence
from typing import NamedTuple

import cupy as cp
import cupy.typing as cpt
import cupyx.scipy.ndimage as ndi
import numpy.typing as npt

KERNEL_PATH = Path(__file__).parent / "kernels" / "brush.cu"
NUDGES = (0.0, 1e-3, 1e-2, 1e-1)  # brush reward perturbations, tried in turn on a stall


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


def _central_adjoint(u: cpt.NDArray, axis: int) -> cpt.NDArray:
    """transpose of `cp.gradient` along `axis`: central inside, one-sided at both ends"""
    n = u.shape[axis]
    first, last = u[_axis_slice(u.ndim, axis, 0)], u[_axis_slice(u.ndim, axis, n - 1)]
    inner = 0.5 * u[_axis_slice(u.ndim, axis, slice(1, n - 1))]
    out = cp.zeros_like(u)
    out[_axis_slice(u.ndim, axis, slice(2, n))] += inner
    out[_axis_slice(u.ndim, axis, slice(0, n - 2))] -= inner
    out[_axis_slice(u.ndim, axis, 1)] += first
    out[_axis_slice(u.ndim, axis, 0)] -= first
    out[_axis_slice(u.ndim, axis, n - 1)] += last
    out[_axis_slice(u.ndim, axis, n - 2)] -= last
    return out


def _fill(s: cpt.NDArray) -> cpt.NDArray:
    """Fill fraction of a disc whose centre sits `s` radii inside the interface."""
    return 0.5 - 15.0 / 16.0 * s + 5.0 / 8.0 * s**3 - 3.0 / 16.0 * s**5


def _fill_slope(s: cpt.NDArray) -> cpt.NDArray:
    """Derivative of `_fill`."""
    return -15.0 / 16.0 + 15.0 / 8.0 * s**2 - 15.0 / 16.0 * s**4


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


class SmoothedProjection(Regularization):
    """`Projection` with its interface averaged over a node, so it survives `beta = inf`.

    Where the `eta` level set of the filtered design passes within `radius` of a node,
    the node takes the fill fraction of the two projected phases a disc of that radius
    sees, read off the distance `(eta - x) / |grad x|` to the interface. The output is
    then binary everywhere but on the interface, whose gradient stays finite at any
    `beta`, so a continuation can run to infinity rather than ending on `threshold`.
    Hammond et al. 2025 (https://arxiv.org/abs/2503.20189), as fdtdx implements it.

    Args:
        beta: sharpness, `math.inf` for the Heaviside itself.
        eta: threshold.
        radius: smoothing radius in nodes, 0.55 as in the reference.
    """

    def __init__(self, beta: float, eta: float = 0.5, radius: float = 0.55) -> None:
        self.beta, self.eta, self.radius = beta, eta, radius

    def _project(self, x: cpt.NDArray) -> tuple[cpt.NDArray, cpt.NDArray]:
        """The plain projection of `x` and its slope, the Heaviside's being zero."""
        if math.isinf(self.beta):
            return (x > self.eta).astype(x.dtype), cp.zeros_like(x)
        projection = Projection(self.beta, self.eta)
        return projection(x), projection.grad(x)

    def _interface(self, x: cpt.NDArray) -> tuple:
        """Gradient components, their norm, the near-interface mask, and `d / radius` there."""
        slopes = cp.gradient(x) if x.ndim > 1 else [cp.gradient(x)]
        norm = cp.sqrt(sum(s * s for s in slopes))
        flat = norm == 0.0
        # a flat node never smooths, so its norm only has to keep the division finite
        norm = cp.where(flat, 1.0, norm)
        scaled = (self.eta - x) / (norm * self.radius)
        near = ~flat & (cp.abs(scaled) < 1.0)
        return slopes, norm, near, cp.where(near, scaled, 0.0)

    def __call__(self, x: cpt.NDArray) -> cpt.NDArray:
        """Projected design, the interface nodes replaced by their fill fraction."""
        projected, _ = self._project(x)
        _, norm, near, s = self._interface(x)
        fill, fill_minus = _fill(s), _fill(-s)
        low, _ = self._project(x - self.radius * norm * fill)
        high, _ = self._project(x + self.radius * norm * fill_minus)
        return cp.where(near, (1.0 - fill) * low + fill * high, projected)

    def grad(self, x: cpt.NDArray, dy: cpt.NDArray | float = 1.0) -> cpt.NDArray:
        """Adjoint: the local chain rule, then the gradient norm's stencil transposed."""
        _, slope = self._project(x)
        slopes, norm, near, s = self._interface(x)
        fill, fill_minus = _fill(s), _fill(-s)
        dfill, dfill_minus = _fill_slope(s), _fill_slope(-s)
        low, low_slope = self._project(x - self.radius * norm * fill)
        high, high_slope = self._project(x + self.radius * norm * fill_minus)
        jump = high - low
        low_slope, high_slope = (1.0 - fill) * low_slope, fill * high_slope
        # d/dx and d/d|grad x| of the smoothed value, s = (eta - x) / (|grad x| R)
        by_value = (
            -jump * dfill / (norm * self.radius)
            + low_slope * (1.0 + dfill)
            + high_slope * (1.0 + dfill_minus)
        )
        by_norm = self.radius * (
            -jump * dfill * s / (norm * self.radius)
            + low_slope * (s * dfill - fill)
            + high_slope * (fill_minus + s * dfill_minus)
        )
        dy = cp.broadcast_to(cp.asarray(dy, dtype=x.dtype), x.shape)
        out = cp.where(near, dy * by_value, dy * slope)
        weight = cp.where(near, dy * by_norm / norm, 0.0)
        for axis, component in enumerate(slopes):
            out += _central_adjoint(weight * component, axis)
        return out


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


def disk(diameter: int) -> cpt.NDArray[cp.bool_]:
    """Circular brush centered on a node, `diameter + 1` nodes across along the axes."""
    offsets = cp.arange(-(diameter // 2), diameter // 2 + 1)
    return cp.hypot(*cp.meshgrid(offsets, offsets, indexing="ij")) <= 0.5 * diameter


class _Touches(NamedTuple):
    """Node and touch states of the brush generator, each a mask over its window."""

    is_solid: cpt.NDArray[cp.bool_]  # painted solid, or fixed solid
    is_void: cpt.NDArray[cp.bool_]
    valid_solid: cpt.NDArray[cp.bool_]  # touches painting no void node
    valid_void: cpt.NDArray[cp.bool_]
    may_solid: cpt.NDArray[cp.bool_]  # nodes some placed or valid touch paints
    may_void: cpt.NDArray[cp.bool_]


class _Stalled(Exception):
    """A brush generator pass left with no touch that strands no node."""


class BrushGenerator(Regularization):
    """A 2D design whose solid and void both admit a disk, by Schubert et al.'s generator.

    Algorithm 1 of https://doi.org/10.1021/acsphotonics.2c00313, each step placing the
    best touch of every neighbourhood three brushes across and dropping any touch that
    would leave a node neither phase can reach. The map is not differentiable, so
    `grad` passes `dy` straight through; docs/regularization.md gives the reward.

    Args:
        diameter: brush diameter in nodes, the least width of either phase.
        region: (nx, ny) mask of the nodes it decides; the rest of `x` is fixed,
            solid above 0.5, and constrains the touches next to it.
    """

    def __init__(self, diameter: int, region: cpt.NDArray[cp.bool_]) -> None:
        if not (isinstance(diameter, int) and diameter >= 1):
            raise ValueError(f"diameter must be a positive integer: {diameter}")
        if region.ndim != 2 or not bool(region.any()):
            raise ValueError(f"region must be a 2D mask with a node: {region.shape}")
        self.brush = disk(diameter)
        self.disk_taps = (cp.argwhere(self.brush) - diameter // 2).astype(cp.int32)
        # a neighbourhood three brushes across, as a row pass then a column pass
        line = cp.arange(-3 * diameter, 3 * diameter + 1, dtype=cp.int32)
        self.row_taps = cp.stack([cp.zeros_like(line), line], axis=1)
        self.column_taps = cp.stack([line, cp.zeros_like(line)], axis=1)
        module = cp.RawModule(code=KERNEL_PATH.read_text())
        self.kernels = {
            name: module.get_function(f"{name}_kernel")
            for name in ("dilate", "maximum", "sum")
        }
        pad = 2 * diameter
        nodes = cp.nonzero(region)
        self.window = tuple(
            slice(max(int(i.min()) - pad, 0), min(int(i.max()) + pad + 1, n))
            for i, n in zip(nodes, region.shape)
        )
        self.free = region[self.window]
        # breaks the ties of a uniform reward, the same way every call
        self.jitter = 1e-6 * cp.random.RandomState(0).random_sample(self.free.shape)
        self.noise = cp.random.RandomState(1).standard_normal(self.free.shape)

    def _launch(self, name: str, a: cpt.NDArray, taps: cpt.NDArray) -> cpt.NDArray:
        """Kernel `name` over the contiguous 2D `a`, combining each node's `taps`."""
        out = cp.empty_like(a)
        n0, n1 = a.shape
        grid = ((n1 + 31) // 32, (n0 + 7) // 8)
        args = (a, out, taps, cp.int32(len(taps)), cp.int32(n0), cp.int32(n1))
        self.kernels[name](grid, (32, 8), args)
        return out

    def _dilate(self, a: cpt.NDArray[cp.bool_]) -> cpt.NDArray[cp.bool_]:
        return self._launch("dilate", a.view(cp.uint8), self.disk_taps).view(cp.bool_)

    def _spread(self, a: cpt.NDArray[cp.bool_]) -> cpt.NDArray[cp.bool_]:
        """`a` dilated by the square neighbourhood a step places one touch in."""
        rows = self._launch("dilate", a.view(cp.uint8), self.row_taps)
        return self._launch("dilate", rows, self.column_taps).view(cp.bool_)

    def _gain(self, reward: cpt.NDArray, painted: cpt.NDArray[cp.bool_]) -> cpt.NDArray:
        """Reward of a touch at each node, summed over the nodes it would newly set.

        A node of the wrong sign weighs as much as the whole brush, so a touch that
        stays inside its phase always wins, and a feasible design comes back as is.
        """
        weighted = cp.where(reward > 0.0, reward, len(self.disk_taps) * reward)
        unset = cp.where(painted, cp.float32(0.0), weighted).astype(cp.float32)
        return self._launch("sum", unset, self.disk_taps)

    def _touches(self, solid, void, fixed_solid, fixed_void) -> _Touches:
        """States of Algorithm 1 for the placed touches, fixed nodes counted as painted."""
        is_solid = self._dilate(solid) | fixed_solid
        is_void = self._dilate(void) | fixed_void
        valid_solid = ~self._dilate(is_void) & ~solid
        valid_void = ~self._dilate(is_solid) & ~void
        may_solid = self._dilate(solid | valid_solid) | fixed_solid
        may_void = self._dilate(void | valid_void) | fixed_void
        return _Touches(is_solid, is_void, valid_solid, valid_void, may_solid, may_void)

    def __call__(self, x: cpt.NDArray) -> cpt.NDArray:
        """`x` with the nodes of `region` set to a feasible 0/1 design."""
        part = x[self.window]
        fixed = (part > 0.5) & ~self.free, (part <= 0.5) & ~self.free
        reward = 2.0 * part - 1.0 + self.jitter
        # a pass can still paint itself into a corner; a nudged reward takes another path
        for nudge in NUDGES:
            try:
                solid = self._generate((reward + nudge * self.noise).astype(cp.float32), fixed)
            except _Stalled:
                continue
            out = x.copy()
            out[self.window] = cp.where(self.free, self._dilate(solid), part)
            return out
        raise RuntimeError(f"the brush generator stalled for every reward nudge {NUDGES}")

    def _generate(self, reward: cpt.NDArray, fixed: tuple) -> cpt.NDArray[cp.bool_]:
        """Solid touches of one pass of Algorithm 1 over the window."""
        solid = cp.zeros(reward.shape, dtype=cp.bool_)
        void = cp.zeros_like(solid)

        # touches first that only paint nodes the other phase cannot reach anyway
        for _ in range(reward.size):
            t = self._touches(solid, void, *fixed)
            if bool(cp.all(t.is_solid | t.is_void)):
                break
            free_solid = ~self._dilate(t.may_void | t.is_void) & t.valid_solid
            free_void = ~self._dilate(t.may_solid | t.is_solid) & t.valid_void
            if bool(cp.any(free_solid | free_void)):
                solid, void = solid | free_solid, void | free_void
                continue

            # near a required node only a touch resolving it, elsewhere any valid one
            need_solid, need_void = ~t.is_solid & ~t.may_void, ~t.is_void & ~t.may_solid
            near = self._spread(need_solid | need_void)
            valid_solid = t.valid_solid & (~near | self._dilate(need_solid))
            valid_void = t.valid_void & (~near | self._dilate(need_void))
            gains = self._gain(reward, t.is_solid), self._gain(-reward, t.is_void)
            solid, void = self._step(solid, void, fixed, valid_solid, valid_void, *gains)
        else:
            raise RuntimeError("the brush generator did not complete the design")
        return solid

    def _step(self, solid, void, fixed, valid_solid, valid_void, gain_solid, gain_void):
        """Place the best touch of each neighbourhood, dropping any that strands a node."""
        while bool(cp.any(valid_solid | valid_void)):
            pick_solid = valid_solid & ((gain_solid >= gain_void) | ~valid_void)
            pick_void = valid_void & ~pick_solid
            score = cp.where(pick_solid, gain_solid, cp.where(pick_void, gain_void, -cp.inf))
            score = score.astype(cp.float32, copy=False)
            rows = self._launch("maximum", score, self.row_taps)
            peak = score == self._launch("maximum", rows, self.column_taps)
            pick_solid, pick_void = peak & pick_solid, peak & pick_void

            # a stranded node is reachable by neither phase; its touch is dropped
            t = self._touches(solid | pick_solid, void | pick_void, *fixed)
            stranded = ~(t.is_solid | t.is_void | t.may_solid | t.may_void)
            if not bool(cp.any(stranded)):
                return solid | pick_solid, void | pick_void
            blamed = self._spread(stranded)
            valid_solid &= ~(blamed & pick_solid)
            valid_void &= ~(blamed & pick_void)
        raise _Stalled

    def grad(self, x: cpt.NDArray, dy: cpt.NDArray | float = 1.0) -> cpt.NDArray:
        """Straight-through estimate: `dy` passed on unchanged."""
        return dy * cp.ones_like(x)


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
