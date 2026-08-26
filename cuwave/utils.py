import itertools

import cupy as cp
import numpy as np

from .sensitivity import l2_misfit, sensitivity
from .wave import Source, simulate

TOL = 1e-6  # for `distribute`, so exact-boundary coordinates survive rounding


# -------------------------------------- general --------------------------------------
def resample(signal, dt, dt_new, N_new=None):
    """Linearly resample `signal` from timestep `dt` to `dt_new`.
    `N_new` defaults to covering the original span; pass it explicitly to force
    a common length across separately resampled signals."""
    xp = cp.get_array_module(signal)
    values = xp.asarray(signal)
    steps = values.shape[0]
    if steps < 2:
        raise ValueError(f"resampling needs at least two samples, got {steps}")
    if N_new is None:
        span = (steps - 1) * dt / dt_new
        N_new = int(span + 1e-9 * max(1.0, span)) + 1

    dtype = xp.dtype(values.dtype if values.dtype.kind == "f" else xp.float64)
    t = xp.arange(N_new, dtype=dtype) * dtype.type(dt_new / dt)  # in input sample index
    left = xp.clip(xp.floor(t), 0, steps - 2)
    weight = t - left
    index = left.astype(xp.int32)
    if values.ndim > 1:
        weight = weight.reshape(N_new, *(1,) * (values.ndim - 1))
    interpolated = (1 - weight) * values[index] + weight * values[index + 1]
    return interpolated * (t.reshape(weight.shape) <= steps - 1)


def line(start, stop, count):
    return np.linspace(start, stop, count)


def distribute(sim, coords):
    """Multilinear interpolation of `coords` onto the grid: (nodes, weights) of the surrounding 2**ndim cell corners."""
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    if coords.shape[1] != sim.ndim:
        raise ValueError(
            f"coordinates need {sim.ndim} components, not {coords.shape[1]}"
        )
    index = np.empty_like(coords)
    for d in range(sim.ndim):
        index[:, d] = coords[:, d] / sim.dx[d] + 1.0
        # checked in index units, where the interior runs from node 1 to Nx[d] - 2
        if index[:, d].min() < 1.0 - TOL or index[:, d].max() > sim.Nx[d] - 2.0 + TOL:
            length = (sim.Nx[d] - 3) * sim.dx[d]  # spanned by the interior nodes
            raise ValueError(
                f"axis {d} coordinate outside the domain [0, {length:g}]: "
                f"[{float(coords[:, d].min()):g}, {float(coords[:, d].max()):g}]"
            )

    base = np.stack(
        [
            np.clip(np.floor(index[:, d]), 1, sim.Nx[d] - 3).astype(np.int32)
            for d in range(sim.ndim)
        ]
    )
    offset = index.T - base  # (ndim, num) position inside that cell, in [0, 1]
    corners = np.array(list(itertools.product((0, 1), repeat=sim.ndim)))  # (K, ndim)
    nodes = base[:, :, None] + corners.T[:, None, :]  # (ndim, num, K)
    weights = np.ones((len(coords), len(corners)))
    for d in range(sim.ndim):
        w = offset[d][:, None]
        weights *= np.where(corners[None, :, d] == 1, w, 1.0 - w)
    return (
        cp.asarray(nodes.reshape(sim.ndim, -1), dtype=cp.int32),
        cp.asarray(weights, dtype=sim.dtype),
    )


def point_source(sim, coords, signal):
    """Build a `Source` injecting `signal` at `coords`, distributed onto grid nodes by `distribute`."""
    nodes, weights = distribute(sim, coords)
    count = weights.shape[0]
    signal = cp.asarray(signal, dtype=sim.dtype)
    if signal.ndim == 1:
        signal = cp.broadcast_to(signal[:, None], (signal.shape[0], count))
    if signal.shape[1] != count:
        raise ValueError(f"{signal.shape[1]} signal columns for {count} sources")
    columns = (signal[:, :, None] * weights).reshape(signal.shape[0], -1)
    return Source(
        nodes, cp.ascontiguousarray(columns / np.prod(sim.dx), dtype=sim.dtype)
    )


def shots(sim, coords, signal):
    """One single-coordinate `Source` per coordinate: the shot list of an inversion"""
    return [point_source(sim, c, signal) for c in np.atleast_2d(coords)]


def stack(sources):
    """Fire several shots in a single simulation.

    Positions and signal columns are concatenated, so a stacked shot costs one
    simulation instead of `len(sources)` and returns one record.

    Encode the shots (random signs, phase shifts) by scaling their signals before stacking.
    """
    return Source(
        cp.concatenate([s.position for s in sources], axis=1),
        cp.ascontiguousarray(cp.concatenate([s.signal for s in sources], axis=1)),
    )


class Sensors:
    """Receivers at arbitrary coordinates, multilinearly interpolated onto the grid."""

    def __init__(self, sim, coords):
        """Precompute the interpolation `nodes` and `weights` for receivers at `coords`."""
        self.sim = sim
        self.coordinates = np.atleast_2d(np.asarray(coords, dtype=float))
        self.count = len(self.coordinates)
        self.nodes, self.weights = distribute(sim, self.coordinates)

    def traces(self, record):
        """(N, count * 2**ndim) node record -> (N, count) receiver traces"""
        return (record.reshape(record.shape[0], self.count, -1) * self.weights).sum(2)

    def scatter(self, dphi):
        """transpose of `traces`: (N, count) -> (N, count * 2**ndim)"""
        return (dphi[:, :, None] * self.weights).reshape(dphi.shape[0], -1)

    def objective(self, objective):
        """Lift a receiver-space `objective` into the node space `sensitivity` wants"""

        def wrapped(record):
            cost, dphi = objective(self.traces(record))
            return cost, self.scatter(dphi)

        return wrapped


# ---------------------------------------- fwi ----------------------------------------
def measure(sim, sources, indicator, sensors):
    """Receiver traces per shot: the synthetic experiment an inversion is fitted to"""
    return [
        sensors.traces(simulate(sim, source, indicator, sensors=sensors.nodes)[1])
        for source in sources
    ]


def misfit(sim, sources, indicator, sensors, observed, objective=l2_misfit):
    """The cost `misfit_gradient` returns, without its gradient: forward passes only."""
    return sum(
        objective(data)(traces)[0]
        for traces, data in zip(measure(sim, sources, indicator, sensors), observed)
    )


def misfit_gradient(sim, sources, indicator, sensors, observed, objective=l2_misfit):
    """Misfit summed over the shots and its derivative with respect to `indicator`.

    Returns `(cost, gradient)`, the gradient a field over the padded grid.
    """
    d_mass, d_stiff = sim.parametrization_jacobian()
    cost = 0.0
    gradient = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    for source, data in zip(sources, observed):
        shot_cost, grads, _ = sensitivity(
            sim, source, indicator, sensors.nodes, sensors.objective(objective(data))
        )
        cost += shot_cost
        gradient += d_mass * grads["mass"] + d_stiff * grads["stiff"]
    return cost, gradient
