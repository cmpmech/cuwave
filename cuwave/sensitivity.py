"""Adjoint sensitivities in four variants, same arguments and same return.

`sensitivity` stores the forward field (N + 2 grids) and is the exact transpose of
the discretisation. `reconstruction_sensitivity` rebuilds that field by a reverse
march instead, storing only the strip a damping layer makes irreversible, and stays
exact wherever the strip shields it. `superposition_sensitivity` reconstructs it in
three slots with no strip at all, trading exactness and a `scale` the caller has to
set for a footprint independent of N. `spectral_sensitivity` keeps only the DFT a
spectral cost reads, exact once its runs decay. All four share the cell weights and
the gradient kernels below; docs/sensitivity.md carries the derivations.

`source_sensitivity` shares those arguments but differentiates with respect to
the source signal rather than the material, which needs no forward field at all.
"""

import math
import warnings
from collections.abc import Callable

import cupy as cp
import cupy.typing as cpt
import cupyx.scipy.ndimage as ndi
import numpy as np
import numpy.typing as npt

from .boundary import define_boundary, pml_nodes
from .wave import (
    BFLOAT16,
    Simulation,
    Source,
    Window,
    compile_kernels,
    define_adjoint_excitation,
    define_excitation,
    define_get_signal,
    define_dft,
    define_set_signal,
    flatten_indices,
    grid_rows,
    march,
    simulate,
)

ADJOINT_DELAY = 1  # lines the adjoint up with the reconstructed forward triplet

# cancellation past which B(w, w) - B(u, u) has eaten too much of the mantissa to trust
CANCELLATION_LIMIT = {"float32": 1e5, "float64": 1e12}

# what `strip_dtype` stores the strip as, per (strip_dtype, precision)
STRIP_DTYPES = {
    ("float32", "float64"): np.dtype(np.float32),
    ("bfloat16", "float32"): BFLOAT16,
    ("bfloat16", "float64"): BFLOAT16,
}

# share of the spectrum the last period still adds, past which the run stopped short
DECAY_LIMIT = 1e-2
# dcost/dtraces off the sinusoids at `frequencies` by more than this is not spectral
FIT_LIMIT = {"float32": 1e-4, "float64": 1e-10}
PULSE_PERIODS = 1.0  # spread of the adjoint pulse envelope, in periods


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


def require_full_domain(sim: Simulation) -> None:
    """Raise if `sim.domain` or `sim.dirichlet` is set: the gradient kernels grade no wall."""
    if sim.masked:
        raise NotImplementedError(
            "the gradient kernels read every cell at the full radius, where a domain "
            "closes some; unset domain, or use source_sensitivity, which needs none"
        )


def require_lossless(sim: Simulation) -> None:
    """Raise if `sim.damping` or `sim.pml` is set: time reversal needs losslessness."""
    if sim.damping is not None or sim.pml is not None:
        raise NotImplementedError(
            "superposition_sensitivity rebuilds the forward field by running it "
            "backwards, which only a lossless operator allows; use sensitivity"
        )


def reconstruction_nodes(
    sim: Simulation,
) -> tuple[cpt.NDArray[cp.int32], cpt.NDArray[cp.bool_]]:
    """Nodes a reverse march has to replay, and where its gradient stays exact.

    Which nodes those are is decided by `sim.damping` and `sim.pml`, so the caller
    states neither.

    Returns:
        (strip, valid): the (ndim, num) grid indices to record and replay each step,
        the damped ones a lossless node reads across the interface, and the mask of
        every lossless interior node, which is where the rebuilt triplet is exact.
    """
    # the reverse step of a node reaches this far, so a strip that thin feeds it
    radius = sim.reach
    interior = cp.zeros(sim.Nx_padded, dtype=cp.bool_)
    interior[tuple(slice(1, n - 1) for n in sim.Nx)] = True
    if sim.damping is None and sim.pml is None:
        return cp.zeros((sim.node_rows, 0), dtype=cp.int32), interior
    lossless = interior & ~pml_nodes(sim)
    if sim.damping is not None:
        lossless &= ~(sim.damping > 0)
    # the full square, as a shear or cross term reads the diagonal neighbours
    square = ndi.generate_binary_structure(sim.ndim, sim.ndim)
    reach = ndi.binary_dilation(
        lossless, structure=square, iterations=radius, brute_force=True
    )
    strip = cp.stack(cp.nonzero(interior & ~lossless & reach)).astype(cp.int32)
    return with_components(sim, strip), lossless


def frozen(sim: Simulation, grads: dict[str, cpt.NDArray]) -> dict[str, cpt.NDArray]:
    """`grads` zeroed in a pml layer, whose memory terms the gradient kernels leave out."""
    if sim.pml is not None:
        layer = pml_nodes(sim)
        # assigned, not multiplied, so a NaN inside cannot survive as 0 * NaN
        for field in grads.values():
            field[..., layer] = 0.0
    return grads


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


def phase_table(sim: Simulation, frequencies: npt.NDArray[np.float64]) -> npt.NDArray:
    """(N, 2F) host table of `cos(omega t)` and `-sin(omega t)`, interleaved per frequency."""
    # the phase reaches 1e5 radians over a long run, so it is built in double
    t = np.arange(sim.N, dtype=np.float64) * sim.dt
    phase = 2.0 * np.pi * t[:, None] * frequencies[None, :]
    table = np.empty((sim.N, 2 * len(frequencies)))
    table[:, 0::2] = np.cos(phase)
    table[:, 1::2] = -np.sin(phase)
    return table


def spectral_load(
    sim: Simulation, dphi: cpt.NDArray, table: npt.NDArray
) -> tuple[npt.NDArray, float]:
    """Fit `dphi` by the sinusoids of `table`: the complex load each frequency drives.

    Returns:
        (load, fit), the (F, num_sensors) `alpha + i beta` with
        `dphi = sum alpha cos + beta sin`, and the relative residual of that fit.
    """
    basis = np.empty_like(table)
    basis[:, 0::2], basis[:, 1::2] = table[:, 0::2], -table[:, 1::2]
    target = cp.asnumpy(dphi).astype(np.float64)
    coef = np.linalg.lstsq(basis, target, rcond=None)[0]
    norm = float(np.linalg.norm(target))
    fit = float(np.linalg.norm(target - basis @ coef)) / norm if norm > 0.0 else 0.0
    return coef[0::2] + 1j * coef[1::2], fit


def spectral_pulse(
    sim: Simulation, frequency: float, load: npt.NDArray
) -> npt.NDArray[np.float64]:
    """(N, num) decaying pulses whose DTFT at `frequency` is `load`, one per column."""
    sigma = PULSE_PERIODS / frequency
    t = np.arange(sim.N, dtype=np.float64) * sim.dt
    if 8.0 * sigma > t[-1]:
        raise ValueError(
            f"the adjoint pulse spans {8.0 * sigma:.3g}, longer than the run {t[-1]:.3g}"
        )
    envelope = np.exp(-0.5 * ((t - 4.0 * sigma) / sigma) ** 2)
    omega = 2.0 * np.pi * frequency
    carriers = envelope * np.stack((np.cos(omega * t), np.sin(omega * t)))
    # each carrier's DTFT at omega, so the pair spans any complex load
    spectra = carriers @ np.exp(-1j * omega * t)
    system = np.array([spectra.real, spectra.imag])
    amplitudes = np.linalg.solve(system, np.stack((load.real, load.imag)))
    return carriers.T @ amplitudes


def settled(
    sim: Simulation, spectrum: cpt.NDArray, last: cpt.NDArray, frequency: float
) -> cpt.NDArray:
    """DFT `spectrum` continued past step N as the constant `last` the run settled to.

    A closed domain keeps the static field a source's net impulse leaves, which never
    decays, so its geometric tail is summed in closed form rather than cut off.
    """
    z = np.exp(-2j * np.pi * frequency * sim.dt)
    return spectrum + last * (z**sim.N / (1.0 - z))


def spectra(
    sim: Simulation, acc: cpt.NDArray, last: cpt.NDArray, frequency: float, steps: int
) -> tuple[cpt.NDArray, float]:
    """One frequency's settled DFT, and the share of it the run's last `steps` added.

    Args:
        sim: the simulation the run stepped.
        acc: (4, num) accumulator: the DFT over every step (real, imaginary), then
            over the last `steps` alone.
        last: (num,) the field the run ended on.
        frequency: the frequency the accumulator was built at.
        steps: how many trailing steps the second pair covers.

    Returns:
        (spectrum, share), the complex (num,) DFT continued past N by `settled`, and
        |tail| / |spectrum| with the constant it settles to taken out of the tail,
        which is what cutting the run at N leaves out, to leading order.
    """
    whole = acc[0].astype(cp.complex128) + 1j * acc[1]
    tail = acc[2].astype(cp.complex128) + 1j * acc[3]
    z = np.exp(-2j * np.pi * frequency * sim.dt)
    tail = tail - last * complex(np.sum(z ** np.arange(sim.N - steps, sim.N)))
    norm = float(cp.linalg.norm(whole))
    share = float(cp.linalg.norm(tail)) / norm if norm > 0.0 else 0.0
    return settled(sim, whole, last, frequency), share


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
    the adjoint backwards is what transposes the damped recursion. A `sim.pml` is
    stepped by its exact transpose, and its layer is taken as frozen material.

    Returns:
        (cost, {"mass": ..., "stiff": ...}, traces, info), the gradients fields over
        the padded grid, zeroed in a pml layer, `traces` the (N, num_sensors) record
        the cost was read from, and `info` empty; this variant has nothing to report.
    """
    require_interior(sim, sensors, "sensor")
    require_interior(sim, source.position, "source")
    require_full_domain(sim)

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, sim.sensitivity_path)

    fd_step = sim.define_step(kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    grads = sim.gradient_fields(mat)

# ------------------------------------ forward pass -----------------------------------
    # stepped straight into the history, so the leading zeros are u^-2 / u^-1
    V = cp.zeros((sim.N + 2, *sim.field_shape), dtype=sim.dtype)

    def forward(t, signal, history):
        u = fd_step(history[t], history[t + 1], history[t + 2])
        excitation_step(u, signal, t)
        bc_step(u)

    march(sim.N, forward, [Window(source.signal), Window(V, halo=2, store=True)])

    # gathered off the history rather than probed per step, saving one launch a step
    um = V[2:].reshape(sim.N, -1)[:, flatten_indices(sim, sensors)]

    cost, dphi = objective(um)

# --------------------------------- adjoint excitation --------------------------------
    signal = adjoint_signal(sim, dphi, sensors)

# ----------------------------------- backward pass -----------------------------------
    P = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    if sim.fuses_adjoint:
        adjoint_step = sim.define_adjoint_gradient(sens_kernels, mat, grads)
        adjoint_excitation = define_adjoint_excitation(sim, sensors, kernels, mat, grads)

        # reversed, so row m is the middle slot both gradients of lambda^(n + 1) read
        def backward(m, signal, history):
            p = adjoint_step(P[m % 2], P[1 - m % 2], history[m])  # lambda^n
            adjoint_excitation(p, signal, history[m], m)
            bc_step(p)

        march(sim.N, backward, [Window(signal), Window(V[::-1])])
        return cost, frozen(sim, sim.finalize_gradients(grads, sens_kernels)), um, {}

    adjoint_step = sim.define_adjoint_step(kernels, sens_kernels, mat)
    gradient_step = sim.define_gradient(sens_kernels, mat, grads)
    adjoint_excitation = define_excitation(sim, sensors, kernels, mat)

    # reversed, so row m + 2 - i is u^(n + i) for n = N - 1 - m
    def backward(m, signal, history):
        p = adjoint_step(P[m % 2], P[1 - m % 2], P[m % 2])  # lambda^n
        adjoint_excitation(p, signal, m)
        bc_step(p)
        # inertia pairs lambda^n with the whole triplet, stiffness with its middle slot
        gradient_step(history[m + 2], history[m + 1], history[m], p)

    march(sim.N, backward, [Window(signal), Window(V[::-1], halo=2)])

    return cost, frozen(sim, sim.finalize_gradients(grads, sens_kernels)), um, {}


def reconstruction_sensitivity(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    objective: Callable,
    strip_dtype: str | None = None,
) -> tuple[float, dict[str, cpt.NDArray], cpt.NDArray, dict]:
    """Cost and its gradients as `sensitivity`, storing a boundary strip and not the field.

    The lossless recursion is symmetric in its two outer slots, so the same step kernel
    run with them swapped marches the forward field backwards. Damping breaks that, so
    the nodes a damped one reaches are recorded each step and replayed on the way back:
    the reverse march then never reads an irreversible node, and the gradient stays the
    exact transpose wherever the strip shields it. This is the variant to reach for
    once the history no longer fits and the domain is open, since
    `superposition_sensitivity` refuses a damping field outright.

    Args:
        sim: the simulation both passes step, damped or lossless. Damping covering
            every interior node is rejected: nothing is left to rebuild from.
        source: the shot to differentiate, its position interior nodes only.
        indicator: the design field the materials are built from.
        sensors: (ndim, num_sensors) interior grid indices.
        objective: takes the (N, num_sensors) record, returns (cost, dcost/dtraces).
        strip_dtype: store the strip narrower than `sim`, halving the memory that
            grows with N at the price of the exact transpose: "float32" under a float64
            sim, or "bfloat16" under either. None keeps it exact. As fdtdx stores its
            boundary record (Mahlau et al. 2026, https://doi.org/10.21105/joss.08912).

    Returns:
        (cost, {"mass": ..., "stiff": ...}, traces, info) as `sensitivity`, the
        gradients zeroed outside the region the strip shields, and `info` carrying the
        `strip` node count and the `drift` the reverse march accumulated.
    """
    require_interior(sim, sensors, "sensor")
    require_interior(sim, source.position, "source")
    require_full_domain(sim)
    strip, valid = reconstruction_nodes(sim)
    require_reconstructable(sim, valid)
    num_strip = strip.shape[1]
    stored = STRIP_DTYPES.get((strip_dtype, sim.precision), sim.dtype)
    if strip_dtype is not None and stored == sim.dtype:
        raise ValueError(
            f"strip_dtype {strip_dtype!r} is not narrower than {sim.precision}: "
            f"use 'bfloat16', or 'float32' under float64"
        )

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, sim.sensitivity_path)

    fd_step = sim.define_step(kernels, mat)
    adjoint_step = sim.define_adjoint_step(kernels, sens_kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    get_signal = define_get_signal(sim, sensors, kernels)
    grads = sim.gradient_fields(mat)
    gradient_step = sim.define_gradient(sens_kernels, mat, grads)
    if num_strip:
        record_strip = define_get_signal(sim, strip, kernels)
        replay_strip = define_set_signal(sim, strip, kernels)

    U = cp.zeros((3, *sim.field_shape), dtype=sim.dtype)
    um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)
    # two leading zero rows, so row t + 2 is u^t and the initial states need no branch
    strip_store = cp.zeros((sim.N + 2, num_strip), dtype=stored)
    windows = [Window(source.signal), Window(um, store=True)]
    if num_strip:
        windows.append(Window(strip_store[2:], store=True, dtype=sim.dtype))

# ------------------------------------ forward pass -----------------------------------
    def forward(t, signal, um, strip=None):
        u = fd_step(U[t % 3], U[(t + 1) % 3], U[(t + 2) % 3])
        excitation_step(u, signal, t)
        bc_step(u)
        get_signal(u, um, t)
        if strip is not None:
            record_strip(u, strip, t)

    march(sim.N, forward, windows)

    cost, dphi = objective(um)

# --------------------------------- adjoint excitation --------------------------------
    signal = adjoint_signal(sim, dphi, sensors)
    adjoint_excitation = define_excitation(sim, sensors, kernels, mat)

# ----------------------------------- backward pass -----------------------------------
    P = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    # the forward rotation left the last three states live, which is the whole seed
    seed = float(cp.linalg.norm(U[(sim.N + 1) % 3] * valid))

    # the triplet (u^(n - 1), u^n, u^(n + 1)) for n = N - 1 - m, rotating backwards
    def triplet(m):
        return tuple(U[(sim.N + i - m) % 3] for i in (2, 0, 1))

    def adjoint(m, signal):
        p = adjoint_step(P[m % 2], P[1 - m % 2], P[m % 2])  # lambda^n
        adjoint_excitation(p, signal, m)
        bc_step(p)
        # inertia pairs lambda^n with the whole triplet, stiffness with its middle slot
        gradient_step(*triplet(m), p)

    def backward(m, signal, source_rows, strip_rows=None):
        adjoint(m, signal)
        a, b, c = triplet(m)
        # read backwards, so the source rides two steps ahead of the state it rebuilds
        c = fd_step(b, a, c)
        excitation_step(c, source_rows, m)
        if strip_rows is not None:
            replay_strip(c, strip_rows, m)
        bc_step(c)

    # row m of each is row n - 1 forwards, contiguous since the kernels index by row
    windows = [Window(signal), Window(source.signal[sim.N - 2 :: -1].copy())]
    if num_strip:
        windows.append(Window(strip_store[sim.N - 2 :: -1].copy(), dtype=sim.dtype))
    march(sim.N - 1, backward, windows)
    adjoint(sim.N - 1, signal)
    a = triplet(sim.N - 1)[0]

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
    require_full_domain(sim)

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, sim.sensitivity_path)

    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    get_signal = define_get_signal(sim, sensors, kernels)
    accs = sim.gradient_fields(mat)
    fused = sim.fuses_adjoint
    if fused:
        subtract_step = sim.define_superposed(sens_kernels, mat, accs, -1.0)
        add_step = sim.define_superposed(sens_kernels, mat, accs, 1.0)
    else:
        fd_step = sim.define_step(kernels, mat)
        subtract_step = sim.define_frechet(sens_kernels, accs, -1.0)
        add_step = sim.define_frechet(sens_kernels, accs, 1.0)

    U = cp.zeros((3, *sim.field_shape), dtype=sim.dtype)
    um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)

# ------------------------------------ forward pass -----------------------------------
    # records the traces and subtracts the forward diagonal B(u, u)
    def forward(t, signal, um):
        u0, u1, u2 = U[t % 3], U[(t + 1) % 3], U[(t + 2) % 3]
        if fused:
            subtract_step(u0, u1, u2)  # the mass density one triplet late
        else:
            fd_step(u0, u1, u2)
        excitation_step(u2, signal, t)
        bc_step(u2)
        get_signal(u2, um, t)
        if not fused:
            subtract_step(u0, u1, u2)

    march(sim.N, forward, [Window(source.signal), Window(um, store=True)])
    if fused:
        t = sim.N - 1
        last = (U[t % 3], U[(t + 1) % 3], U[(t + 2) % 3])
        sim.define_frechet(sens_kernels, accs, -1.0, stiffness=False)(*last)

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
    # starting from u^(N - 1) over u^(N - 2), so the one array carries u + k lambda
    def backward(t, signal, row, mass=True):
        u0, u1, u2 = (U[(sim.N + i - t) % 3] for i in (1, 0, 2))
        if fused:
            add_step(u0, u1, u2, mass)
        else:
            fd_step(u0, u1, u2)
        backward_excitation(u2, signal, row)
        bc_step(u2)
        if not fused:
            add_step(u0, u1, u2)

    def shifted(t, signal):
        backward(t + 1, signal, t)

    # outside the graph, as the first fused step's triplet before is the forward's last
    backward(0, backward_signal, 0, mass=False)
    march(sim.N - 1, shifted, [Window(backward_signal[1:])])
    if fused:
        last = (U[(1 + i) % 3] for i in (1, 0, 2))
        sim.define_frechet(sens_kernels, accs, 1.0, stiffness=False)(*last)

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


def spectral_sensitivity(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32],
    objective: Callable,
    frequencies: npt.ArrayLike,
    region: cpt.NDArray[cp.bool_] | None = None,
) -> tuple[float, dict[str, cpt.NDArray], cpt.NDArray, dict]:
    """Cost and its gradients as `sensitivity`, for a cost read off the record's spectrum.

    A cost that sees the traces only through their DFT at `frequencies` (`intensity`)
    has a gradient pairing the forward and adjoint fields at those frequencies alone,
    so both passes run forwards in time and keep a running DFT instead of a history.
    The adjoint is driven by a decaying pulse carrying the same spectral load, which a
    damped operator allows, so this is the variant for an open domain whose cost is
    spectral. It is exact once both runs have decayed; `info["decay"]` measures how far
    they have. Adapted from fdtdx (Mahlau et al. 2026,
    https://doi.org/10.21105/joss.08912) and the Meep adjoint (Hammond et al. 2022,
    https://doi.org/10.1364/OE.442074); docs/sensitivity.md derives it.

    Args:
        sim: the simulation both passes step, damped or lossless.
        source: the shot to differentiate, its position interior nodes only.
        indicator: the design field the materials are built from.
        sensors: (ndim, num_sensors) interior grid indices.
        objective: takes the (N, num_sensors) record, returns (cost, dcost/dtraces),
            the derivative a sum of sinusoids at `frequencies`.
        frequencies: the frequencies the cost reads, one adjoint run each.
        region: nodal mask of where the gradient is wanted, or None for everywhere.
            The DFT is only kept around it, which on a small design region makes it
            nearly free.

    Returns:
        (cost, {"mass": ..., "stiff": ...}, traces, info) as `sensitivity`, the
        gradients zeroed outside `region`, and `info` carrying the `decay` both runs
        left and the `fit` residual of the spectral load.
    """
    require_interior(sim, sensors, "sensor")
    require_interior(sim, source.position, "source")
    require_full_domain(sim)
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))

    logical = cp.zeros(sim.Nx_padded, dtype=cp.bool_)
    logical[tuple(slice(0, n) for n in sim.Nx)] = True
    if region is None:
        region = logical
    # the gradient at a node reads its neighbours, ghosts included, so the DFT does too
    square = ndi.generate_binary_structure(sim.ndim, sim.ndim)
    kept = ndi.binary_dilation(
        region, structure=square, iterations=sim.reach + 1, brute_force=True
    )
    nodes = cp.stack(cp.nonzero(kept & logical)).astype(cp.int32)
    nodes = with_components(sim, nodes)
    lin_index = flatten_indices(sim, nodes)

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, sim.sensitivity_path)

    fd_step = sim.define_step(kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    adjoint_excitation = define_excitation(sim, sensors, kernels, mat)
    get_signal = define_get_signal(sim, sensors, kernels)
    grads = sim.gradient_fields(mat)
    gradient_step = sim.define_gradient(sens_kernels, mat, grads)

    host_table = phase_table(sim, frequencies)
    period = math.ceil(1.0 / (float(frequencies.min()) * sim.dt))
    # the same factors again over the last period alone, for how far the run settled
    tail = host_table.copy()
    tail[: sim.N - period] = 0.0
    table = cp.asarray(np.concatenate((host_table, tail), axis=1), dtype=sim.dtype)
    F = len(frequencies)
    u_hat = cp.zeros((4 * F, nodes.shape[1]), dtype=sim.dtype)
    forward_dft = define_dft(sim, nodes, kernels, u_hat)
    U = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)

# ------------------------------------ forward pass -----------------------------------
    def forward(t, signal, um, table):
        u = fd_step(U[t % 2], U[1 - t % 2], U[t % 2])
        excitation_step(u, signal, t)
        bc_step(u)
        get_signal(u, um, t)
        forward_dft(u, table, t)

    windows = [Window(source.signal), Window(um, store=True), Window(table)]
    march(sim.N, forward, windows)
    u_last = U[(sim.N - 1) % 2].ravel()[lin_index].astype(cp.float64)

    cost, dphi = objective(um)

# --------------------------------- adjoint excitation --------------------------------
    load, fit = spectral_load(sim, dphi, host_table)
    if fit > FIT_LIMIT[sim.precision]:
        raise ValueError(
            f"dcost/dtraces is off the sinusoids at {frequencies} by {fit:.1e}: "
            f"the cost is not spectral there, use sensitivity"
        )
    weights = cp.asnumpy(sim.adjoint_weights(sensors)).astype(np.float64)

# ----------------------------------- backward pass -----------------------------------
    # each frequency drives a causal run, whose DFT is the adjoint at that frequency
    lam_hat = cp.zeros((4, nodes.shape[1]), dtype=sim.dtype)
    adjoint_dft = define_dft(sim, nodes, kernels, lam_hat)
    fields = cp.zeros((4, *sim.field_shape), dtype=sim.dtype)
    decay = 0.0

    def adjoint(t, signal, table):
        p = adjoint_step(U[t % 2], U[1 - t % 2], U[t % 2])
        adjoint_excitation(p, signal, t)
        bc_step(p)
        adjoint_dft(p, table, t)

    for f, frequency in enumerate(frequencies):
        signal = spectral_pulse(sim, frequency, load[f]) / weights
        signal = cp.ascontiguousarray(cp.asarray(signal, dtype=sim.dtype))
        U[:] = 0.0
        lam_hat[:] = 0.0
        # defined per run, so a pml's memory starts from rest each time
        adjoint_step = sim.define_adjoint_step(kernels, sens_kernels, mat)
        columns = [2 * f, 2 * f + 1, 2 * F + 2 * f, 2 * F + 2 * f + 1]
        march(sim.N, adjoint, [Window(signal), Window(table[:, columns].copy())])
        lam_last = U[(sim.N - 1) % 2].ravel()[lin_index].astype(cp.float64)

        spectrum, share = spectra(sim, u_hat[columns], u_last, frequency, period)
        adjoint_hat, adjoint_share = spectra(sim, lam_hat, lam_last, frequency, period)
        decay = max(decay, share, adjoint_share)
        # lambda^t pairs with (u^(t - 2), u^(t - 1), u^t), so the triplet is z^-2, z^-1, 1
        shift = np.exp(-2j * np.pi * frequency * sim.dt)
        triplet = (spectrum * shift**2, spectrum * shift, spectrum)
        # Re(a b) = Re a Re b - Im a Im b, the two halves of the bilinear form
        for part, sign in ((cp.real, 1.0), (cp.imag, -1.0)):
            for k, value in enumerate((*triplet, sign * adjoint_hat)):
                fields[k].ravel()[lin_index] = part(value)
            gradient_step(*fields)

    grads = frozen(sim, sim.finalize_gradients(grads, sens_kernels))
    # assigned, not multiplied, so a NaN outside cannot survive as 0 * NaN
    outside = ~region
    for field in grads.values():
        field[..., outside] = 0.0
    if decay > DECAY_LIMIT:
        warnings.warn(
            f"the last period still adds more than {DECAY_LIMIT:.0e} of the spectrum: "
            f"the gradient misses the tail the run cut off (info['decay'] says how "
            f"much). Lengthen the run or damp the faces.",
            RuntimeWarning,
            stacklevel=2,
        )
    info = {"decay": decay, "fit": fit}
    return cost, grads, um, info


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

# ------------------------------------ forward pass -----------------------------------
    um = simulate(sim, source, indicator, sensors)[1]
    cost, dphi = objective(um)

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, sim.sensitivity_path)
    adjoint_step = sim.define_adjoint_step(kernels, sens_kernels, mat)
    bc_step = define_boundary(sim, kernels)
    probe = define_get_signal(sim, source.position, kernels)

# --------------------------------- adjoint excitation --------------------------------
    signal = adjoint_signal(sim, dphi, sensors)
    adjoint_excitation = define_excitation(sim, sensors, kernels, mat)

# ----------------------------------- backward pass -----------------------------------
    P = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    lam = cp.zeros((sim.N, source.position.shape[1]), dtype=sim.dtype)

    def backward(m, signal, lam):
        p = adjoint_step(P[m % 2], P[1 - m % 2], P[m % 2])  # lambda^n
        adjoint_excitation(p, signal, m)
        bc_step(p)
        probe(p, lam, m)

    march(sim.N, backward, [Window(signal), Window(lam, store=True)])

    # the transpose of adjoint_signal: over the same weights, and reversed back
    gradient = lam[::-1] * sim.adjoint_weights(source.position)
    return cost, gradient, um, {}
