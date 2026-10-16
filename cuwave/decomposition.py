"""Axis 0 of one grid split over several GPUs, every subdomain an ordinary `Simulation`.

A subdomain steps its own rows plus `halo_rows` of each neighbour's with the unmodified
kernels. The wall its cut makes spreads an error inward by `sim.reach` rows a step, and
`exchange` refreshes the halo from the owners every `EVERY` steps, before it arrives.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import cupy as cp
import cupy.typing as cpt
import numpy as np

from .boundary import _canonical_faces, pml_widths

if TYPE_CHECKING:
    from .wave import Simulation

EVERY = 6  # steps between exchanges, a multiple of the slot rotation periods 2 and 3

_context = threading.local()


@dataclass
class Team:
    devices: tuple[int, ...]
    bounds: list[tuple[int, int, int, int]]  # global rows (g0, g1, o0, o1): held, owned
    home: int  # the device the caller's arrays live on
    barrier: threading.Barrier = None
    posts: list = field(default_factory=list)  # what each subdomain published last
    shared: object = None  # the leader's answer to a collective

    def __post_init__(self) -> None:
        self.barrier = threading.Barrier(len(self.devices))
        self.posts = [None] * len(self.devices)


# -------------------------------------- helpers --------------------------------------
def halo_rows(sim: Simulation) -> int:
    """Rows held past a cut: two exchange intervals of reach, then the gradient's read."""
    return 2 * EVERY * sim.reach + sim.space_order + 2


def stale_rows(sim: Simulation) -> int:
    """Halo rows one exchange interval can spoil: its reach past the cut wall's own stencil."""
    return EVERY * sim.reach + sim.space_order + 2


def partition(sim: Simulation) -> list[tuple[int, int, int, int]]:
    """(g0, g1, o0, o1) per device: the rows a subdomain holds and those it owns."""
    n, rows, halo = len(sim.devices), sim.Nx[0], halo_rows(sim)
    if sim.ndim < 2:
        raise ValueError("a decomposition splits axis 0, which a 1D grid pads")
    edges = np.linspace(0, rows, n + 1).round().astype(int)
    if np.diff(edges).min() < halo:
        raise ValueError(f"{rows} rows over {n} devices own fewer than the {halo} halo")
    lo, hi = pml_widths(sim)[0] if sim.pml is not None else (0, 0)
    if lo and edges[1] - halo <= lo or hi and edges[-2] + halo >= rows - 1 - hi:
        raise ValueError(
            f"a halo of {halo} rows reaches into the axis-0 pml of {lo, hi}"
        )
    return [
        (max(0, a - halo), min(rows, b + halo), a, b)
        for a, b in zip(edges[:-1], edges[1:])
    ]


def _on(device: int):
    """`device` made current, on its legacy stream unless it is the thread's own."""
    if getattr(_context, "device", None) in (None, device):
        return cp.cuda.Device(device)
    # one current stream per thread: a subdomain's own would not match another device
    stack = contextlib.ExitStack()
    stack.enter_context(cp.cuda.Device(device))
    stack.enter_context(cp.cuda.Stream.null)
    return stack


def _rows(array: cpt.NDArray, axis: int, a: int, b: int) -> cpt.NDArray:
    """Rows `a .. b` of `array` along `axis`, as a view."""
    return array[(slice(None),) * axis + (slice(a, b),)]


def _move(array, device: int) -> cpt.NDArray:
    """`array` (host or any device) on `device`, contiguous, shared if it already is."""
    if isinstance(array, np.ndarray):
        with _on(device):
            return cp.asarray(array)
    if array.device.id == device and array.flags.c_contiguous:
        return array
    with _on(array.device.id):
        packed = cp.ascontiguousarray(array)
        cp.cuda.get_current_stream().synchronize()
    if packed.device.id == device:
        return packed
    with _on(device):
        out = cp.empty(packed.shape, dtype=packed.dtype)
        # on the stream the pool recycles `out` from, so behind whatever still reads it
        stream = cp.cuda.get_current_stream()
        out.data.copy_from_device_async(packed.data, packed.nbytes, stream)
        stream.synchronize()
    return out


def _views(sim: Simulation, state: list) -> dict:
    """The arrays of `state` keyed alike on every subdomain, each with its axis-0 axis.

    A field ends in `sim.Nx_padded`; anything else is a pml memory, one per lined axis
    in `pml_slabs` order. An axis-0 slab never reaches a halo, so it is left out.
    """
    slabs = sim.pml_slabs() if sim.pml is not None else []
    padded = sim.Nx_padded
    views, fields, memories = {}, 0, 0
    for array in state:
        if array.shape[-sim.ndim :] == padded:
            views["field", fields] = (array, array.ndim - sim.ndim)
            fields += 1
            continue
        if not slabs or array.ndim != 2:
            raise NotImplementedError(
                f"cannot split a {array.shape} memory over devices"
            )
        d, lo, hi = slabs[memories % len(slabs)][:3]
        key = ("slab", d, memories // len(slabs))
        memories += 1
        if d == 0:
            continue
        if d == sim.ndim - 1:
            views[key] = (array.reshape(len(array), *padded[:-1], lo + hi), 1)
        else:
            rest = [padded[e] for e in range(sim.ndim) if e != d]
            views[key] = (array.reshape(len(array), lo + hi, *rest), 2)
    return views


def active() -> bool:
    """Whether this thread steps a subdomain, so `march` has to exchange."""
    return getattr(_context, "team", None) is not None


def exchange(state: list) -> None:
    """Overwrite every halo row of `state` with the owner's, on every subdomain at once."""
    team = getattr(_context, "team", None)
    if team is None:
        return
    k, sim = _context.index, _context.sim
    views = _views(sim, state)
    cp.cuda.get_current_stream().synchronize()
    team.posts[k] = views
    team.barrier.wait()
    g0, g1, o0, o1 = team.bounds[k]
    device = team.devices[k]
    depth = halo_rows(sim) if getattr(_context, "deep", False) else stale_rows(sim)
    for j, (a, b) in (
        (k - 1, (g0, min(o0, g0 + depth))),
        (k + 1, (max(o1, g1 - depth), g1)),
    ):
        if b <= a:
            continue
        offset = team.bounds[j][0]
        for key, (array, axis) in views.items():
            source, _ = team.posts[j][key]
            halo = _move(_rows(source, axis, a - offset, b - offset), device)
            _rows(array, axis, a - g0, b - g0)[...] = halo
    cp.cuda.get_current_stream().synchronize()
    team.barrier.wait()


@contextlib.contextmanager
def replaying():
    """Exchange whole halos inside: a replayed strip brings back what its recording held stale."""
    _context.deep = True
    try:
        yield
    finally:
        _context.deep = False


def agree(value, reduce: Callable = min):
    """`reduce` of `value` over every subdomain, so their schedules stay in step."""
    team = getattr(_context, "team", None)
    if team is None:
        return value
    team.posts[_context.index] = value
    team.barrier.wait()
    result = reduce(team.posts)
    team.barrier.wait()
    return result


# ------------------------------------- subdomains ------------------------------------
def _subdomain(sim: Simulation, k: int, team: Team) -> Simulation:
    """The `Simulation` subdomain `k` steps: its rows, its arrays, its share of the pml."""
    g0, g1 = team.bounds[k][:2]
    device = team.devices[k]
    arrays = {}
    for name in ("damping", "domain", "dirichlet"):
        value = getattr(sim, name)
        arrays[name] = None if value is None else _move(value[g0:g1], device)
    pml = sim.pml
    if pml is not None:
        cut = {0} if k else set()
        cut |= {1} if k < len(team.devices) - 1 else set()
        faces = tuple(f for f in _canonical_faces(sim.ndim, pml.faces) if f not in cut)
        pml = replace(pml, faces=faces) if faces else None
    with cp.cuda.Device(device):
        return replace(sim, Nx=(g1 - g0, *sim.Nx[1:]), pml=pml, devices=None, **arrays)


def _points(sim: Simulation, local: Simulation, position, k: int, team: Team) -> tuple:
    """Columns of `position` subdomain `k` holds, theirs in local rows, and those it owns."""
    g0, g1, o0, o1 = team.bounds[k]
    axis = sim.node_rows - sim.ndim
    row = position[axis]
    held = np.nonzero((row >= g0 + 1) & (row <= g1 - 2))[0]
    owned = (row[held] >= o0) & (row[held] < o1)
    local_position = position[:, held]
    local_position[axis] -= g0
    if not len(held):
        # an idle column at an interior node, so every launch keeps its shape
        local_position = np.ones((sim.node_rows, 1), dtype=np.int32)
        local_position[axis] = (o0 + o1) // 2 - g0
    with _on(team.devices[k]):
        local_position = cp.asarray(local_position, dtype=cp.int32)
    return local_position, held, owned


def _columns(record, held, k: int, team: Team) -> cpt.NDArray:
    """Columns `held` of a home `record` on subdomain `k`, one idle zero column if none."""
    device = team.devices[k]
    if not len(held):
        with _on(device):
            return cp.zeros((record.shape[0], 1), dtype=record.dtype)
    with _on(record.device.id):
        block = record[:, cp.asarray(held)]
    return _move(block, device)


def _gather(parts: list, columns: list, total: int, home: int) -> cpt.NDArray:
    """One (N, total) home record from the owned columns of every subdomain's record."""
    with _on(home):
        rows = parts[0].shape[0]
        record = cp.zeros((rows, total), dtype=parts[0].dtype)
        for part, (held, owned) in zip(parts, columns):
            if owned.any():
                with _on(part.device.id):
                    block = part[:, cp.asarray(np.nonzero(owned)[0])]
                record[:, cp.asarray(held[owned])] = _move(block, home)
    return record


def _stitch(parts: list, sim: Simulation, team: Team, host: bool):
    """One global field from the owned rows of each subdomain's, along axis 0."""
    axis = parts[0].ndim - sim.ndim
    blocks = [
        _rows(part, axis, o0 - g0, o1 - g0)
        for part, (g0, _, o0, o1) in zip(parts, team.bounds)
    ]
    if host:
        return np.concatenate([cp.asnumpy(b) for b in blocks], axis=axis)
    with _on(team.home):
        return cp.concatenate([_move(b, team.home) for b in blocks], axis=axis)


# ---------------------------------------- runner -------------------------------------
def decomposable(function: Callable) -> Callable:
    """Run `function` split over `sim.devices` when they are set, unchanged otherwise.

    `function` is `simulate` or a sensitivity variant: it takes `(sim, source,
    indicator, sensors, ...)`, and the variants an `objective`, which sees the whole
    record once, on the caller's device. Outputs come back there too, or onto the host
    when `indicator` is a host array, which is what lets a grid outgrow one device.
    """
    signature = inspect.signature(function)

    @functools.wraps(function)
    def run(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        sim = bound.arguments["sim"]
        if sim.devices is None:
            return function(*args, **kwargs)
        if bound.arguments.get("record_every") is not None:
            raise NotImplementedError("record_every is not split over devices")
        return _run(function, sim, bound.arguments)

    return run


def undivided(function: Callable) -> Callable:
    """Refuse `sim.devices` for a `function` that has no split of its own yet."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def run(*args, **kwargs):
        if signature.bind(*args, **kwargs).arguments["sim"].devices is not None:
            raise NotImplementedError(
                f"{function.__name__} runs on one device: unset devices"
            )
        return function(*args, **kwargs)

    return run


def _run(function: Callable, sim: Simulation, arguments: dict):
    """Each subdomain of `sim` in a thread of its own, the outputs stitched together."""
    cp.cuda.get_current_stream().synchronize()
    team = Team(tuple(sim.devices), partition(sim), cp.cuda.Device().id)
    source, indicator = arguments["source"], arguments["indicator"]
    sensors, objective = arguments.get("sensors"), arguments.get("objective")
    positions = cp.asnumpy(source.position)
    nodes = None if sensors is None else cp.asnumpy(sensors)
    host = isinstance(indicator, np.ndarray)
    columns = [None] * len(team.devices)

    def work(k):
        device = team.devices[k]
        g0, g1 = team.bounds[k][:2]
        with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True) as stream:
            _context.device = device
            try:
                local = _subdomain(sim, k, team)
                position, held, _ = _points(sim, local, positions, k, team)
                signal = _columns(source.signal, held, k, team)
                local_args = {**arguments, "sim": local}
                local_args["source"] = replace(source, position=position, signal=signal)
                local_args["indicator"] = _move(indicator[g0:g1], device)
                if sensors is not None:
                    local_args["sensors"], held, owned = _points(
                        sim, local, nodes, k, team
                    )
                    columns[k] = (held, owned)
                if objective is not None:
                    local_args["objective"] = _objective(
                        objective, k, team, columns, sensors
                    )
                _context.team, _context.index, _context.sim = team, k, local
                out = function(**local_args)
                stream.synchronize()
            except BaseException:
                # released, so no other subdomain waits for this one at a barrier
                team.barrier.abort()
                raise
            finally:
                _context.team, _context.device = None, None
        return out

    with ThreadPoolExecutor(len(team.devices)) as pool:
        futures = [pool.submit(work, k) for k in range(len(team.devices))]
    errors = [f.exception() for f in futures if f.exception() is not None]
    if errors:
        # the subdomain that failed first, not the others its abort released
        raise min(errors, key=lambda e: isinstance(e, threading.BrokenBarrierError))
    outs = [f.result() for f in futures]
    total = 0 if sensors is None else sensors.shape[1]
    if objective is None:
        fields = _stitch(
            [o[0] if isinstance(o, tuple) else o for o in outs], sim, team, host
        )
        if sensors is None:
            return fields
        return fields, _gather([o[1] for o in outs], columns, total, team.home)
    cost = outs[0][0]
    grads = {
        name: _stitch([o[1][name] for o in outs], sim, team, host)
        for name in outs[0][1]
    }
    traces = _gather([o[2] for o in outs], columns, total, team.home)
    info = {}
    for key in outs[0][3]:
        values = [o[3][key] for o in outs]
        info[key] = sum(values) if key == "strip" else max(values)
    return cost, grads, traces, info


def _objective(objective: Callable, k: int, team: Team, columns: list, sensors):
    """`objective` evaluated once on the whole record, each subdomain handed its columns."""

    def local(um):
        cp.cuda.get_current_stream().synchronize()
        team.posts[k] = um
        team.barrier.wait()
        if k == 0:
            record = _gather(team.posts, columns, sensors.shape[1], team.home)
            with _on(team.home):
                team.shared = objective(record)
                cp.cuda.get_current_stream().synchronize()
        team.barrier.wait()
        cost, dphi = team.shared
        held = columns[k][0]
        return cost, _columns(dphi, held, k, team).astype(um.dtype, copy=False)

    return local
