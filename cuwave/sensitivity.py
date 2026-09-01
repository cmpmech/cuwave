"""Adjoint sensitivities in three variants, same arguments and same return.

`sensitivity` stores the forward field (N + 2 grids) and is the exact transpose of
the discretisation. `reconstruction_sensitivity` rebuilds that field by a reverse
march instead, storing only the strip a damping layer makes irreversible, and stays
exact wherever the strip shields it. `superposition_sensitivity` reconstructs it in
three slots with no strip at all, trading exactness and a `scale` the caller has to
set for a footprint independent of N. All three share the cell weights and the
adjoint excitation below; docs/sensitivity.md carries the derivations.

`source_sensitivity` shares those arguments but differentiates with respect to
the source signal rather than the material, which needs no forward field at all.
"""

import warnings
from collections.abc import Callable

import cupy as cp
import cupy.typing as cpt
import cupyx.scipy.ndimage as ndi

from .boundary import define_boundary
from .wave import (
    Simulation,
    Source,
    compile_kernels,
    define_excitation,
    define_get_signal,
    define_set_signal,
    define_step_method,
    flatten_indices,
    grid_rows,
)

ADJOINT_DELAY = 1  # lines the adjoint up with the reconstructed forward triplet

# cancellation past which B(w, w) - B(u, u) has eaten too much of the mantissa to trust
CANCELLATION_LIMIT = {"float32": 1e5, "float64": 1e12}


# -------------------------------------- helpers --------------------------------------
def windowed_misfit(observed: cpt.NDArray, window: cpt.NDArray) -> Callable:
    """Objective factory: `l2_misfit` restricted to a time window.

    Args:
        observed: the (N, num_sensors) measured record to fit.
        window: (N, 1) or (N, num_sensors) weights, zero outside the window kept.

    Returns:
        the objective `sensitivity` takes, its derivative carrying the window twice
        so that it stays the exact derivative of the windowed cost.
    """

    def objective(traces):
        residual = window * (traces - observed)
        return 0.5 * float(cp.sum(residual**2)), window * residual

    return objective


def _accumulated(accs: dict) -> float:
    """Norm over every accumulator, so the diagnostic names no material field."""
    return float(sum(float(cp.linalg.norm(f)) ** 2 for f in accs.values()) ** 0.5)


def l2_misfit(observed: cpt.NDArray) -> Callable:
    """Objective factory: J = 1/2 sum (traces - observed)^2, and its derivative."""

    def objective(traces):
        residual = traces - observed
        return 0.5 * float(cp.sum(residual**2)), residual

    return objective


def require_interior(
    sim: Simulation, position: cpt.NDArray[cp.int32], what: str
) -> None:
    """Raise if any node of `position` sits on a ghost node, naming it `what`."""
    # a ghost node carries no equation, so a sensor there corrupts the whole gradient
    rows = grid_rows(sim, position)
    lo = int(cp.min(rows))
    if lo < 1:
        raise ValueError(f"{what} on a ghost node: index {lo} < 1")
    for d in range(sim.ndim):
        hi = int(cp.max(rows[d]))
        if hi > sim.Nx[d] - 2:
            raise ValueError(
                f"{what} on a ghost node: axis {d} index {hi} exceeds "
                f"Nx[{d}] - 2 = {sim.Nx[d] - 2}"
            )


def require_lossless(sim: Simulation) -> None:
    """Raise if `sim.damping` is set: reconstructing by time reversal needs losslessness."""
    if sim.damping is not None:
        raise NotImplementedError(
            "superposition_sensitivity rebuilds the forward field by running it "
            "backwards, which only a lossless operator allows; use sensitivity"
        )


def reconstruction_nodes(
    sim: Simulation,
) -> tuple[cpt.NDArray[cp.int32], cpt.NDArray[cp.bool_]]:
    """Nodes a reverse march has to replay, and where its gradient stays exact.

    Which nodes those are is decided by `sim.damping`, so the caller states neither.

    Returns:
        (strip, valid): the (ndim, num) grid indices to record and replay each step,
        the damped ones a lossless node reads across the interface, and the mask of
        every lossless interior node, which is where the rebuilt triplet is exact.
    """
    # the reverse step of a node reaches this far, so a strip that thin feeds it
    radius = sim.reach
    interior = cp.zeros(sim.Nx_padded, dtype=cp.bool_)
    interior[tuple(slice(1, n - 1) for n in sim.Nx)] = True
    if sim.damping is None:
        return cp.zeros((sim.node_rows, 0), dtype=cp.int32), interior
    lossless = interior & ~(sim.damping > 0)
    reach = ndi.binary_dilation(lossless, iterations=radius, brute_force=True)
    strip = cp.stack(cp.nonzero(interior & ~lossless & reach)).astype(cp.int32)
    return with_components(sim, strip), lossless


def with_components(
    sim: Simulation, nodes: cpt.NDArray[cp.int32]
) -> cpt.NDArray[cp.int32]:
    """Repeat spatial `nodes` once per field component, the component row prepended."""
    if sim.ncomp == 1:
        return nodes
    tiled = cp.tile(nodes, (1, sim.ncomp))
    row = cp.repeat(cp.arange(sim.ncomp, dtype=cp.int32), nodes.shape[1])
    return cp.ascontiguousarray(cp.concatenate((row[None, :], tiled), axis=0))


def require_reconstructable(sim: Simulation, valid: cpt.NDArray[cp.bool_]) -> None:
    """Raise if `sim.damping` leaves no lossless interior for a reverse march to rebuild."""
    if not bool(cp.any(valid)):
        raise ValueError(
            "damping covers every interior node, so there is nothing to reconstruct; "
            "damp only the faces with boundary.sponge, or use sensitivity"
        )


def adjoint_signal(
    sim: Simulation,
    dphi: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    scale: float = 1.0,
    delay: int = 0,
) -> cpt.NDArray:
    """Adjoint excitation: `dphi` reversed, over W and source_factor, times `scale`.

    Args:
        dphi: (N, num_sensors) derivative of the cost with respect to the traces.
        sensors: the nodes it is injected on, one column each.
        scale: the superposition k; 1 for the exact adjoint.
        delay: entries dropped from the front and zero-padded at the back, which
            starts the adjoint recursion that many steps earlier in its own sequence.
    """
    # define_excitation supplies the dt^2 source_factor minv the recursion wants
    signal = cp.asarray(dphi, dtype=sim.dtype)[::-1] / sim.adjoint_weights(sensors)
    if delay:
        signal = cp.concatenate((signal[delay:], cp.zeros_like(signal[:delay])))
    return cp.ascontiguousarray(sim.dtype(scale) * signal, dtype=sim.dtype)


# ---------------------------------- adjoint solvers ----------------------------------
def sensitivity(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    objective: Callable,
) -> tuple[float, dict[str, cpt.NDArray], cpt.NDArray, dict]:
    """Cost and its gradients d(cost)/d(mass, stiff) over the padded grid.

    Args:
        sim: the simulation the forward and adjoint passes both step.
        source: the shot to differentiate, its position interior nodes only.
        indicator: the design field the materials are built from.
        sensors: (ndim, num_sensors) interior grid indices.
        objective: takes the (N, num_sensors) record, returns (cost, dcost/dtraces).
            The derivative drives the adjoint field, so any differentiable cost
            works; reparametrize by chain rule at the call site with
            `sim.parametrization_jacobian()`.

    A `sim.damping` field is stepped by the same kernel in both passes, since marching
    the adjoint backwards is what transposes the damped recursion.

    Returns:
        (cost, {"mass": ..., "stiff": ...}, traces, info), the gradients fields over
        the padded grid, `traces` the (N, num_sensors) record the cost was read from,
        and `info` empty; this variant has nothing to report.
    """
    require_interior(sim, sensors, "sensor")
    require_interior(sim, source.position, "source")

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, sim.sensitivity_path)

    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    grads = sim.gradient_fields(mat)
    gradient_step = sim.define_gradient(sens_kernels, mat, grads)

    # ------------------------------------ forward pass -----------------------------------
    # stepped straight into the history, so the leading zeros are u^-2 / u^-1
    V = cp.zeros((sim.N + 2, *sim.field_shape), dtype=sim.dtype)
    # the slot views made once: V[t] is a host slice costing more than its own kernel
    slot = [V[t] for t in range(sim.N + 2)]

    for t in range(sim.N):
        u = fd_step(slot[t], slot[t + 1], slot[t + 2])
        u = excitation_step(u, source.signal, t)
        u = bc_step(u)

    # gathered off the history rather than probed per step, saving one launch a step
    um = V[2:].reshape(sim.N, -1)[:, flatten_indices(sim, sensors)]

    cost, dphi = objective(um)

    # --------------------------------- adjoint excitation --------------------------------
    signal = adjoint_signal(sim, dphi, sensors)
    adjoint_excitation = define_excitation(sim, sensors, kernels, mat)

    # ----------------------------------- backward pass -----------------------------------
    P = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    p0, p1 = P[0], P[1]
    for m in range(sim.N):
        n = sim.N - 1 - m
        p0 = fd_step(p0, p1, p0)
        p0 = adjoint_excitation(p0, signal, m)
        p0 = bc_step(p0)
        p1, p0 = p0, p1  # p1 now holds lambda^n
        # inertia pairs lambda^n with the whole triplet, stiffness with its middle slot
        gradient_step(slot[n], slot[n + 1], slot[n + 2], p1)

    return cost, sim.finalize_gradients(grads, sens_kernels), um, {}


def reconstruction_sensitivity(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    objective: Callable,
) -> tuple[float, dict[str, cpt.NDArray], cpt.NDArray, dict]:
    """Cost and its gradients as `sensitivity`, storing a boundary strip and not the field.

    The lossless recursion is symmetric in its two outer slots, so the same step kernel
    run with them swapped marches the forward field backwards. Damping breaks that, so
    the nodes a damped one reaches are recorded each step and replayed on the way back:
    the reverse march then never reads an irreversible node, and the gradient stays the
    exact transpose wherever the strip shields it. This is the variant to reach for once
    the history no longer fits and the domain is open, since `superposition_sensitivity`
    refuses a damping field outright.

    Args:
        sim: the simulation both passes step, damped or lossless. Damping covering
            every interior node is rejected: nothing is left to rebuild from.
        source: the shot to differentiate, its position interior nodes only.
        indicator: the design field the materials are built from.
        sensors: (ndim, num_sensors) interior grid indices.
        objective: takes the (N, num_sensors) record, returns (cost, dcost/dtraces).

    Returns:
        (cost, {"mass": ..., "stiff": ...}, traces, info) as `sensitivity`, the
        gradients zeroed outside the region the strip shields, and `info` carrying the
        `strip` node count and the `drift` the reverse march accumulated.
    """
    require_interior(sim, sensors, "sensor")
    require_interior(sim, source.position, "source")
    strip, valid = reconstruction_nodes(sim)
    require_reconstructable(sim, valid)
    num_strip = strip.shape[1]

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, sim.sensitivity_path)

    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    get_signal = define_get_signal(sim, sensors, kernels)
    grads = sim.gradient_fields(mat)
    gradient_step = sim.define_gradient(sens_kernels, mat, grads)
    if num_strip:
        record_strip = define_get_signal(sim, strip, kernels)
        replay_strip = define_set_signal(sim, strip, kernels)

    U = cp.zeros((3, *sim.field_shape), dtype=sim.dtype)
    u0, u1, u2 = U[0], U[1], U[2]
    um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)
    # two leading zero rows, so row t + 2 is u^t and the initial states need no branch
    strip_store = cp.zeros((sim.N + 2, num_strip), dtype=sim.dtype)

    # ------------------------------------ forward pass -----------------------------------
    for t in range(sim.N):
        u2 = fd_step(u0, u1, u2)
        u2 = excitation_step(u2, source.signal, t)
        u2 = bc_step(u2)
        get_signal(u2, um, t)
        if num_strip:
            record_strip(u2, strip_store, t + 2)
        u0, u1, u2 = u1, u2, u0

    cost, dphi = objective(um)

    # --------------------------------- adjoint excitation --------------------------------
    signal = adjoint_signal(sim, dphi, sensors)
    adjoint_excitation = define_excitation(sim, sensors, kernels, mat)

    # ----------------------------------- backward pass -----------------------------------
    P = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    p0, p1 = P[0], P[1]
    # the forward rotation left the last three states live, which is the whole seed
    a, b, c = u2, u0, u1
    seed = float(cp.linalg.norm(c * valid))

    for m in range(sim.N):
        n = sim.N - 1 - m
        p0 = fd_step(p0, p1, p0)
        p0 = adjoint_excitation(p0, signal, m)
        p0 = bc_step(p0)
        p1, p0 = p0, p1  # p1 now holds lambda^n
        # inertia pairs lambda^n with the whole triplet, stiffness with its middle slot
        gradient_step(a, b, c, p1)
        if n < 1:
            break
        # read backwards, so the source rides two steps ahead of the state it rebuilds
        c = fd_step(b, a, c)
        c = excitation_step(c, source.signal, n - 1)
        if num_strip:
            replay_strip(c, strip_store, n - 1)
        c = bc_step(c)
        a, b, c = c, a, b

    grads = sim.finalize_gradients(grads, sens_kernels)
    # assigned, not multiplied, so a NaN outside cannot survive as 0 * NaN
    outside = ~valid
    for field in grads.values():
        field[outside] = 0.0
    # the march ends on the initial state, which is zero, so what is left is round-off
    drift = float(cp.linalg.norm(a * valid)) / seed if seed > 0.0 else float("inf")
    info = {"strip": num_strip, "drift": drift}
    return cost, grads, um, info


def superposition_sensitivity(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    objective: Callable,
    scale: float = 1.0,
) -> tuple[float, dict[str, cpt.NDArray], cpt.NDArray, dict]:
    """Cost and its gradients as `sensitivity`, in three field slots instead of N + 2.

    The gradient densities are bilinear in the forward and adjoint fields, so the
    cross term can be read off the diagonal of B(u + k lambda) alone, and the closed
    lossless domain is time-reversible, so u itself never has to be stored. The prices
    are a k^2 bias traded against round-off, and a symmetric Frechet form that is
    consistent with `sensitivity` rather than equal to it. docs/sensitivity.md has
    both, with the measurements.

    Args:
        sim: the simulation both passes step, which must be lossless: a `damping`
            field is rejected, since the time reversal needs one. `sensitivity`
            takes one.
        source: the shot to differentiate, its position interior nodes only.
        indicator: the design field the materials are built from.
        sensors: (ndim, num_sensors) interior grid indices.
        objective: takes the (N, num_sensors) record, returns (cost, dcost/dtraces).
        scale: the superposition k, trading the k**2 bias against round-off. Set it
            from `info["cancellation"]`, aiming near 1e4 in float32 or 1e6 in float64.

    Returns:
        (cost, {"mass": ..., "stiff": ...}, traces, info) as `sensitivity`, with
        `info` carrying `scale` and the `cancellation` the subtraction cost.
    """
    require_lossless(sim)
    require_interior(sim, sensors, "sensor")
    require_interior(sim, source.position, "source")

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, sim.sensitivity_path)

    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    get_signal = define_get_signal(sim, sensors, kernels)
    accs = sim.gradient_fields(mat)
    subtract_step = sim.define_frechet(sens_kernels, accs, -1.0)
    add_step = sim.define_frechet(sens_kernels, accs, 1.0)

    U = cp.zeros((3, *sim.field_shape), dtype=sim.dtype)
    u0, u1, u2 = U[0], U[1], U[2]
    um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)

    # ------------------------------------ forward pass -----------------------------------
    # records the traces and subtracts the forward diagonal B(u, u)
    for t in range(sim.N):
        u2 = fd_step(u0, u1, u2)
        u2 = excitation_step(u2, source.signal, t)
        u2 = bc_step(u2)
        get_signal(u2, um, t)
        subtract_step(u0, u1, u2)
        u0, u1, u2 = u1, u2, u0

    cost, dphi = objective(um)

    # --------------------------------- adjoint excitation --------------------------------
    # the forward diagonal before the backward pass cancels it, for `cancellation`
    before = _accumulated(accs)
    # concatenated into one launch, sound because excitation_kernel uses atomicAdd
    backward_position = cp.concatenate((sensors, source.position), axis=1)
    backward_signal = cp.ascontiguousarray(
        cp.concatenate(
            (
                adjoint_signal(sim, dphi, sensors, scale, ADJOINT_DELAY),
                source.signal[: sim.N][::-1],
            ),
            axis=1,
        )
    )
    backward_excitation = define_excitation(sim, backward_position, kernels, mat)

    # ----------------------------------- backward pass -----------------------------------
    # u0 / u1 hold u^(N-1) / u^(N-2), so the one array carries u + k lambda
    u0, u1 = u1, u0
    for t in range(sim.N):
        u2 = fd_step(u0, u1, u2)
        u2 = backward_excitation(u2, backward_signal, t)
        u2 = bc_step(u2)
        add_step(u0, u1, u2)
        u0, u1, u2 = u1, u2, u0

    after = _accumulated(accs)
    cancellation = before / after if after > 0.0 else float("inf")
    limit = CANCELLATION_LIMIT[sim.precision]
    if cancellation > limit:
        warnings.warn(
            f"superposition scale={scale:g} leaves a cancellation of "
            f"{cancellation:.1e} in {sim.precision}, past the usable {limit:.0e}: the "
            f"gradient is largely round-off. Raise scale by about "
            f"{cancellation / limit:.0e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # B(u, lambda) = [B(w, w) - B(u, u)] / 2k, scaled in place to spare a field
    norm = sim.dtype(1.0 / (2.0 * scale))
    accs = sim.finalize_gradients(accs, sens_kernels)
    for field in accs.values():
        field *= norm
    info = {"scale": scale, "cancellation": cancellation}
    return cost, accs, um, info


def source_sensitivity(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    objective: Callable,
) -> tuple[float, cpt.NDArray, cpt.NDArray, dict]:
    """Cost and its gradient d(cost)/d(source.signal), the adjoint field at the source.

    Args:
        sim: the simulation the forward and adjoint passes both step.
        source: the shot to differentiate, its position interior nodes only.
        indicator: the design field the materials are built from, held fixed here.
        sensors: (ndim, num_sensors) interior grid indices.
        objective: takes the (N, num_sensors) record, returns (cost, dcost/dtraces).

    The cost is linear in the signal, so the gradient pairs no forward field against
    the adjoint one and this variant stores neither: four grids, whatever N is.

    Returns:
        (cost, gradient, traces, info), the gradient the (N, num_sources) derivative
        with respect to `source.signal`, `traces` the (N, num_sensors) record the cost
        was read from, and `info` empty; this variant has nothing to report.
    """
    require_interior(sim, sensors, "sensor")
    require_interior(sim, source.position, "source")

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)

    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    get_signal = define_get_signal(sim, sensors, kernels)
    probe = define_get_signal(sim, source.position, kernels)

    # ------------------------------------ forward pass -----------------------------------
    U = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    u0, u1 = U[0], U[1]
    um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)

    for t in range(sim.N):
        u0 = fd_step(u0, u1, u0)
        u0 = excitation_step(u0, source.signal, t)
        u0 = bc_step(u0)
        u1, u0 = u0, u1
        get_signal(u1, um, t)

    cost, dphi = objective(um)

    # --------------------------------- adjoint excitation --------------------------------
    signal = adjoint_signal(sim, dphi, sensors)
    adjoint_excitation = define_excitation(sim, sensors, kernels, mat)

    # ----------------------------------- backward pass -----------------------------------
    P = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    p0, p1 = P[0], P[1]
    lam = cp.zeros((sim.N, source.position.shape[1]), dtype=sim.dtype)

    for m in range(sim.N):
        n = sim.N - 1 - m
        p0 = fd_step(p0, p1, p0)
        p0 = adjoint_excitation(p0, signal, m)
        p0 = bc_step(p0)
        p1, p0 = p0, p1  # p1 now holds lambda^n
        probe(p1, lam, n)  # row n rather than row m, so the record runs forward in time

    # the transpose of adjoint_signal: over the same weights, and not reversed
    gradient = lam * sim.adjoint_weights(source.position)
    return cost, gradient, um, {}
