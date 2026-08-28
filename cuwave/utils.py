import itertools
from collections.abc import Callable, Sequence

import cupy as cp
import cupy.typing as cpt
import numpy as np
import numpy.typing as npt

from .sensitivity import l2_misfit, sensitivity
from .wave import Simulation, Source, simulate

TOL = 1e-6  # for `distribute`, so exact-boundary coordinates survive rounding


# -------------------------------------- helpers --------------------------------------
def _reparametrize(sim: Simulation, grads: dict[str, cpt.NDArray]) -> cpt.NDArray:
    """Chain d(cost)/d(mass, stiff) onto the indicator with `parametrization_jacobian`."""
    d_mass, d_stiff = sim.parametrization_jacobian()
    return d_mass * grads["mass"] + d_stiff * grads["stiff"]


# -------------------------------------- general --------------------------------------
def interior_slice(sim: Simulation) -> tuple[slice, ...]:
    """Index tuple selecting the interior nodes, dropping the ghost ring and the padding."""
    return tuple(slice(1, n - 1) for n in sim.Nx)


def threshold(
    field: cpt.NDArray,
    eta: float = 0.5,
    low: float = 0.0,
    high: float = 1.0,
    dtype: npt.DTypeLike | None = None,
) -> cpt.NDArray:
    """Snap a grey design to `low` below the threshold `eta` and `high` above it.

    `dtype` defaults to the dtype of `field`, since `cp.where` against python floats
    otherwise promotes a float32 design to float64 and the kernels read it as garbage.
    """
    snapped = cp.where(field < eta, low, high)
    if dtype is None:
        dtype = field.dtype if field.dtype.kind == "f" else snapped.dtype
    return snapped.astype(dtype)


def resample(
    signal: cpt.NDArray | npt.NDArray,
    dt: float,
    dt_new: float,
    N_new: int | None = None,
) -> cpt.NDArray | npt.NDArray:
    """Linearly resample `signal` from timestep `dt` to `dt_new`.

    Args:
        signal: samples along the leading axis, on either array module.
        dt: the timestep `signal` is sampled at.
        dt_new: the timestep to resample onto.
        N_new: number of output samples, defaulting to the original span. Pass it
            explicitly to force a common length across separately resampled signals.

    Returns:
        the resampled signal, zero past the end of the original span.
    """
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


def line(
    start: npt.ArrayLike, stop: npt.ArrayLike, count: int
) -> npt.NDArray[np.float64]:
    """`count` coordinates evenly spaced from `start` to `stop`, endpoints included."""
    return np.linspace(start, stop, count)


def distribute(
    sim: Simulation, coords: npt.ArrayLike
) -> tuple[cpt.NDArray[cp.int32], cpt.NDArray]:
    """Multilinear interpolation of `coords` onto the grid.

    Args:
        sim: the simulation whose grid the coordinates land on.
        coords: (num, ndim) physical coordinates, inside the domain.

    Returns:
        (nodes, weights) of the surrounding 2**ndim cell corners, shaped
        (ndim, num * 2**ndim) and (num, 2**ndim).
    """
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


def point_source(
    sim: Simulation, coords: npt.ArrayLike, signal: cpt.NDArray | npt.NDArray
) -> Source:
    """Build a `Source` injecting `signal` at `coords`, distributed by `distribute`.

    Args:
        sim: the simulation the source fires into.
        coords: (num, ndim) physical coordinates of the point sources.
        signal: one (N,) trace broadcast to every coordinate, or (N, num) one each.

    Returns:
        the `Source`, its signal divided by the cell volume so the amplitude is
        independent of the grid spacing.
    """
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


def collect_source(
    sim: Simulation, coords: npt.ArrayLike, columns: cpt.NDArray
) -> cpt.NDArray:
    """Transpose of `point_source`: an (N, num * 2**ndim) node gradient onto `coords`.

    Args:
        sim: the simulation the columns were injected into.
        coords: (num, ndim) physical coordinates the source was built from.
        columns: the derivative with respect to the node columns `point_source` made.

    Returns:
        the (N, num) derivative with respect to the signal of each coordinate.
    """
    _, weights = distribute(sim, coords)
    columns = columns.reshape(columns.shape[0], len(weights), -1)
    # cast, since dividing a float32 record by a float64 cell volume would promote it
    return (columns * weights).sum(2) / sim.dtype(np.prod(sim.dx))


def shots(
    sim: Simulation, coords: npt.ArrayLike, signal: cpt.NDArray | npt.NDArray
) -> list[Source]:
    """One single-coordinate `Source` per coordinate: the shot list of an inversion"""
    return [point_source(sim, c, signal) for c in np.atleast_2d(coords)]


def stack(sources: Sequence[Source]) -> Source:
    """Fire several shots in a single simulation.

    Positions and signal columns are concatenated, so a stacked shot costs one
    simulation instead of `len(sources)` and returns one record.

    Encode the shots (random signs, phase shifts) by scaling their signals first.
    """
    return Source(
        cp.concatenate([s.position for s in sources], axis=1),
        cp.ascontiguousarray(cp.concatenate([s.signal for s in sources], axis=1)),
    )


class Sensors:
    """Receivers at arbitrary coordinates, multilinearly interpolated onto the grid.

    The interpolation `nodes` and `weights` are precomputed once, so `traces` and its
    transpose `scatter` are the only per-record work.

    Args:
        sim: the simulation whose grid the receivers land on.
        coords: (count, ndim) physical coordinates of the receivers.
    """

    def __init__(self, sim: Simulation, coords: npt.ArrayLike) -> None:
        self.sim = sim
        self.coordinates = np.atleast_2d(np.asarray(coords, dtype=float))
        self.count = len(self.coordinates)
        self.nodes, self.weights = distribute(sim, self.coordinates)

    def traces(self, record: cpt.NDArray) -> cpt.NDArray:
        """(N, count * 2**ndim) node record -> (N, count) receiver traces"""
        return (record.reshape(record.shape[0], self.count, -1) * self.weights).sum(2)

    def scatter(self, dphi: cpt.NDArray) -> cpt.NDArray:
        """transpose of `traces`: (N, count) -> (N, count * 2**ndim)"""
        return (dphi[:, :, None] * self.weights).reshape(dphi.shape[0], -1)

    def objective(self, objective: Callable) -> Callable:
        """Lift a receiver-space `objective` into the node space `sensitivity` wants"""

        def wrapped(record):
            cost, dphi = objective(self.traces(record))
            return cost, self.scatter(dphi)

        return wrapped


# ---------------------------------------- fwi ----------------------------------------
def measure(
    sim: Simulation,
    sources: Sequence[Source],
    indicator: cpt.NDArray,
    sensors: Sensors,
) -> list[cpt.NDArray]:
    """Receiver traces per shot: the synthetic experiment an inversion is fitted to"""
    return [
        sensors.traces(simulate(sim, source, indicator, sensors=sensors.nodes)[1])
        for source in sources
    ]


def misfit(
    sim: Simulation,
    sources: Sequence[Source],
    indicator: cpt.NDArray,
    sensors: Sensors,
    observed: Sequence[cpt.NDArray],
    objective: Callable = l2_misfit,
) -> float:
    """The cost `misfit_gradient` returns, without its gradient: forward passes only."""
    return sum(
        objective(data)(traces)[0]
        for traces, data in zip(measure(sim, sources, indicator, sensors), observed)
    )


def misfit_gradient(
    sim: Simulation,
    sources: Sequence[Source],
    indicator: cpt.NDArray,
    sensors: Sensors,
    observed: Sequence[cpt.NDArray],
    objective: Callable = l2_misfit,
    adjoint: Callable = sensitivity,
) -> tuple[float, cpt.NDArray]:
    """Misfit summed over the shots and its derivative with respect to `indicator`.

    Args:
        sim: the simulation each shot is run in.
        sources: the shot list, one adjoint solve each.
        indicator: the design field the gradient is taken with respect to.
        sensors: the receiver array the objective is evaluated on.
        observed: the measured traces, one (N, count) record per shot.
        objective: factory taking one record and returning `objective(traces)`.
        adjoint: which variant computes each shot, `reconstruction_sensitivity`
            where the stored history no longer fits.

    Returns:
        (cost, gradient), the gradient a field over the padded grid, already
        reparametrized from (mass, stiff) onto `indicator`.
    """
    cost = 0.0
    gradient = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    for source, data in zip(sources, observed):
        shot_cost, grads, _, _ = adjoint(
            sim, source, indicator, sensors.nodes, sensors.objective(objective(data))
        )
        cost += shot_cost
        gradient += _reparametrize(sim, grads)
    return cost, gradient


# ---------------------------------------- tato ---------------------------------------
def energy(sim: Simulation) -> Callable:
    """Objective factory: J = 1/2 int_region int_t p^2, the energy reaching the sensors."""
    scale = float(np.prod(sim.dx)) * sim.dt

    def objective(traces):
        return 0.5 * scale * float(cp.sum(traces**2)), scale * traces

    return objective


def response(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    objective: Callable,
) -> tuple[float, cpt.NDArray]:
    """The cost `response_gradient` returns, without its gradient: one forward pass.

    Returns:
        (cost, wavefield), the field at the last step over the logical grid, so the
        design a cost was read from is plotted together with the wave that scored it.
    """
    wavefield, traces = simulate(sim, source, indicator, sensors=sensors)
    return objective(traces)[0], wavefield


def response_gradient(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    objective: Callable,
    adjoint: Callable = sensitivity,
) -> tuple[float, cpt.NDArray]:
    """Cost of one shot and its derivative with respect to `indicator`.

    Args:
        sim: the simulation the forward and adjoint passes both step.
        source: the shot to differentiate, its position interior nodes only.
        indicator: the design field the gradient is taken with respect to.
        sensors: (ndim, num_sensors) interior grid indices the objective reads.
        objective: takes the (N, num_sensors) record, returns (cost, dcost/dtraces).
        adjoint: which variant computes the gradient, `reconstruction_sensitivity`
            where the stored history no longer fits.

    Returns:
        (cost, gradient), the gradient a field over the padded grid, already
        reparametrized from (mass, stiff) onto `indicator`.
    """
    cost, grads, _, _ = adjoint(sim, source, indicator, sensors, objective)
    return cost, _reparametrize(sim, grads)
