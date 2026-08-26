from pathlib import Path

import cupy as cp

from .boundary import define_boundary
from .wave import (
    Source,
    compile_kernels,
    define_excitation,
    define_step_method,
    flatten_indices,
    grid_block,
)

KERNEL_PATH = Path(__file__).parent / "kernels" / "wave_sensitivity.cu"


def l2_misfit(observed):
    # objective factory: J = 1/2 sum (traces - observed)^2 and its derivative
    def objective(traces):
        residual = traces - observed
        return 0.5 * float(cp.sum(residual**2)), residual

    return objective


# A node on a wall owns half a cell per boundary axis, and the adjoint identity is
# L^T = W L W^-1 with W those cell volumes -- so the exact discrete gradient carries
# W, and the adjoint source carries 1/W. Without them the wall ring comes out
# exactly 2x per boundary axis too large (4x in a corner), which a finite-difference
# check catches at once.
#
# W is never materialised. It is {1, 1/2, 1/4}, so as a field it would be one more
# full grid held for the whole run -- 1 of the 8 the superposition variant lives on.
# The two things it is needed for both avoid that: scaling a gradient is a slice-wise
# in-place multiply, and the adjoint source only ever needs W at the sensor nodes.
#
# W is derived for the mirrored-ghost Neumann wall, but it needs no Dirichlet variant:
# there the wall layer is zero in the forward field and, the adjoint recursion being the
# same step, in the adjoint field too -- so whatever W scales on that layer is zero
# either way, and the exactness above survives unchanged. Pinned by
# test_dirichlet_wall_gradient_matches_finite_differences.


def apply_cell_weights(sim, field):
    # in-place multiply by W. Axes in sequence, so a corner compounds to 1/4 (1/8 in
    # 3D) exactly as the product over its boundary axes should.
    for d in range(sim.ndim):
        for index in (1, sim.Nx[d] - 2):
            face = [slice(None)] * sim.ndim
            face[d] = index
            field[tuple(face)] *= 0.5
    return field


def sensor_cell_weights(sim, sensors):
    # W at the sensor nodes only, as a (num_sensors,) vector. Written as the same
    # nested loop as apply_cell_weights rather than a membership test, so the two
    # agree node for node even on a degenerate axis where Nx[d] - 2 == 1.
    w = cp.ones(sensors.shape[1], dtype=sim.dtype)
    for d in range(sim.ndim):
        for index in (1, sim.Nx[d] - 2):
            w = cp.where(sensors[d] == index, w * 0.5, w)
    return w


def require_interior(sim, position, what):
    # A ghost node is a slaved mirror, not an unknown: it carries no equation, so a
    # sensor sitting on one injects the objective derivative into nothing and
    # silently corrupts the gradient over the whole grid -- not just nearby. Cheap
    # to check once, and the off-by-one is easy to write (Nx[d] - 1 is the ghost).
    lo = int(cp.min(position))
    if lo < 1:
        raise ValueError(f"{what} on a ghost node: index {lo} < 1")
    for d in range(sim.ndim):
        hi = int(cp.max(position[d]))
        if hi > sim.Nx[d] - 2:
            raise ValueError(
                f"{what} on a ghost node: axis {d} index {hi} exceeds "
                f"Nx[{d}] - 2 = {sim.Nx[d] - 2}"
            )


def index_geometry(sim):
    # kernel args after the fields, for a kernel with no per-axis factor:
    # N0, [N1, s0], [N2, s1] -- the factor-carrying variant is wave.axis_geometry
    geom = [sim.Nx[0]]
    for d in range(1, sim.ndim):
        geom += [sim.Nx[d], sim.strides[d - 1]]
    return geom


def define_gradient(sim, kernels, mat):
    # One kernel for both gradients, and a prebuilt argument list mutated in place:
    # see the launch-cost note above define_step_method in wave.py, which is why
    # this is shaped the way it is rather than one closure per material field.
    gradient_kernel = kernels.get_function("gradient_kernel")
    grid, block = grid_block(sim)
    # the operator without the dt^2 the step folds into it: L, not dt^2 L
    factors = [sim.dtype(float(f) / sim.dt**2) for f in sim.step_factors()]
    geom = [factors[0], sim.Nx[0]]
    for d in range(1, sim.ndim):
        geom += [factors[d], sim.Nx[d], sim.strides[d - 1]]
    args = [None] * 6 + [mat["stiff"], sim.dtype(1.0 / sim.dt**2), *geom]

    def gradient_step(g_mass, g_stiff, u0, u1, u2, l1):
        args[0], args[1] = g_mass, g_stiff
        args[2], args[3], args[4], args[5] = u0, u1, u2, l1
        gradient_kernel(grid, block, args)

    return gradient_step


def sensitivity(sim, source, indicator, sensors, objective, damping=None):
    """Cost and its gradients d(cost)/d(mass, stiff) over the padded grid.

    `objective(traces)` takes the (N, num_sensors) sensor record and returns
    (cost, dcost/dtraces). The derivative is what drives the adjoint field, so
    any differentiable cost works -- reparametrize by chain rule at the call site
    with `sim.parametrization_jacobian()`.

    Returns (cost, {"mass": ..., "stiff": ...}, traces).
    """
    if damping is not None:
        raise NotImplementedError(
            "the adjoint of a damped step is not that step run backwards; the "
            "reverse-time pass here needs a lossless, self-adjoint operator"
        )

    require_interior(sim, sensors, "sensor")
    require_interior(sim, source.position, "source")

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, KERNEL_PATH)

    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    gradient_step = define_gradient(sim, sens_kernels, mat)

# --------------------------------- forward pass ----------------------------------
    # stepped straight into the history buffer, so the two leading zero slots are
    # u^-2 / u^-1 and no copy is needed: u^n lives in V[n + 2]
    V = cp.zeros((sim.N + 2, *sim.Nx_padded), dtype=sim.dtype)
    # the slot views, made once: V[t] is a host-side slice costing ~1.5 us, which is
    # more than the kernel it feeds, and each pass below takes three or four of them
    # per step. The list is N + 2 view objects, a few hundred bytes each.
    slot = [V[t] for t in range(sim.N + 2)]

    for t in range(sim.N):
        u = fd_step(slot[t], slot[t + 1], slot[t + 2])
        u = excitation_step(u, source.signal, t)
        u = bc_step(u)

    # the traces are read off the history in one gather rather than probed per step:
    # `simulate` needs a get_signal launch each step because it keeps only two slots,
    # but here every field is still on the device, and one launch saved per step is
    # a quarter of this pass (see the launch-cost note in wave.py). The sensors are
    # interior nodes by require_interior, so the ghost mirroring never touches them
    # and reading after the loop gives the same values as reading inside it.
    um = V[2:].reshape(sim.N, -1)[:, flatten_indices(sim, sensors)]

    cost, dphi = objective(um)

# ------------------------------- adjoint excitation -------------------------------
    # define_excitation multiplies by dt^2 * source_factor * minv, which is exactly
    # the m the adjoint recursion wants, so the signal carries the remaining
    # 1 / (W source_factor) -- and time is reversed. The 1/W is the same cell volume
    # that scales the gradients below: a sensor on a wall owns half a cell and
    # without it drives the adjoint field at half strength (a quarter in a corner),
    # which is a plain factor 2 on the whole gradient for a single wall sensor, and
    # for a mixed wall/interior array a gradient that is not even wrong by a constant.
    w_sensors = sensor_cell_weights(sim, sensors)
    signal = cp.ascontiguousarray(
        cp.asarray(dphi, dtype=sim.dtype)[::-1] / (w_sensors * sim.source_factor()),
        dtype=sim.dtype,
    )
    adjoint = Source(sensors, signal)
    adjoint_excitation = define_excitation(sim, adjoint.position, kernels, mat)

# --------------------------------- backward pass ---------------------------------
    P = cp.zeros((2, *sim.Nx_padded), dtype=sim.dtype)
    p0, p1 = P[0], P[1]
    g_mass = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    g_stiff = cp.zeros(sim.Nx_padded, dtype=sim.dtype)

    for m in range(sim.N):
        n = sim.N - 1 - m
        p0 = fd_step(p0, p1, p0)
        p0 = adjoint_excitation(p0, adjoint.signal, m)
        p0 = bc_step(p0)
        p1, p0 = p0, p1  # p1 now holds lambda^n
        # the inertia term pairs lambda^n with the whole u^n .. u^{n-2} triplet,
        # the stiffness term with u^{n-1} alone -- the operator in the residual is
        # evaluated one step back, which is the middle slot of that same triplet
        gradient_step(g_mass, g_stiff, slot[n], slot[n + 1], slot[n + 2], p1)

    apply_cell_weights(sim, g_mass)
    apply_cell_weights(sim, g_stiff)
    return cost, {"mass": g_mass, "stiff": g_stiff}, um
