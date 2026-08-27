import math
import time

import cupy as cp
import cupy.typing as cpt
import matplotlib.pyplot as plt
import numpy as np

from cuwave.evals import l2_error
from cuwave.sensitivity import l2_misfit, sensitivity, superposition_sensitivity
from cuwave.signals import ricker
from cuwave.utils import point_source
from cuwave.wave import ScalarWave, grid_coords, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 2  # fixed
PRECISION = "float64"  # float32 floors the relative error near 1e-7
THREADS = (4, 128)
# "standard" keeps the forward history -- N + 2 grids, which is what caps REFERENCE
# below -- and is the exact transpose; "superposition" runs in three slots instead and
# lifts that cap, at the price of a consistent-not-exact gradient and a scale to pick
METHOD = "standard"  # "standard" | "superposition"
SUPERPOSITION_SCALE = 1e6  # aim for a cancellation near 1e6 in float64

# discretization
SPACE_ORDERS = (2, 4, 6, 8)
# every level has to divide REFERENCE, so that the reference restricts onto it by
# strided slicing, and be a multiple of SENSOR_DIVISOR, so that the sensors land on
# nodes at the same physical points on all of them
LEVELS = (32, 48, 64, 96, 128, 192)  # cells per axis
REFERENCE = 384  # cells per axis: 2x the finest level, and a multiple of every one
REFERENCE_SAFETY = 0.5  # the reference refines in time as well as in space
REFERENCE_ORDER = 12
SAFETY = 0.99  # fraction of the stable time step, which every run takes in full

# physics
LENGTH = 1.0
WAVESPEED = 1.0
DENSITY = 1.0
DENSITY0 = 1e-4  # inside the inclusion
T = 2.0  # the scattered arrival is back at the sensors by 1.0

# source
AMPLITUDE = 1.0
FREQUENCY = 4.0  # 8 points per wavelength on the coarsest level, 48 on the finest
SENSOR_DIVISOR = 16  # sensors at k * LENGTH / this along the left edge, k = 1 .. 15

# geometry
RADIUS = 0.1
# a staircased inclusion is only O(dx)-consistent across the levels and would cap
# every order there; a finite transition width resolves on all of them. 0.0 is the
# sharp disk of examples/sensitivity/scalar_sensitivity_2D.py
SMOOTHING = 0.02

# postprocessing
COLORS = ("k", "b", "g", "r")  # one per entry of SPACE_ORDERS

# --------------------------------------- setup ---------------------------------------
if any(REFERENCE % n_el for n_el in LEVELS):
    raise ValueError(f"every level must divide REFERENCE={REFERENCE}: {LEVELS}")
if any(n_el % SENSOR_DIVISOR for n_el in LEVELS):
    raise ValueError(f"every level must be a multiple of {SENSOR_DIVISOR}: {LEVELS}")
if len(COLORS) < len(SPACE_ORDERS):
    raise ValueError(f"{len(COLORS)} colors for {len(SPACE_ORDERS)} space orders")

mempool = cp.get_default_memory_pool()

print(
    f"{WAVESPEED / (FREQUENCY * LENGTH / min(LEVELS)):.0f} points per wavelength "
    f"on the coarsest level"
)


# --------------------------------------- helper --------------------------------------
def gradient_of(
    space_order: int, n_el: int, dt_stable: float
) -> tuple[cpt.NDArray, float, int]:
    """Sensitivity of the misfit at the homogeneous model, on `n_el` cells per axis.

    The inclusion is measured and inverted at the same level, as in the sensitivity
    driver: the objective is that level's own discrete misfit, and it converges to the
    continuous one along with everything else.

    Args:
        space_order: finite difference order of the forward and adjoint passes.
        n_el: cells per axis, so the grid holds `n_el + 3` nodes including ghosts.
        dt_stable: upper bound on the timestep, which `T` is then divided into.

    Returns:
        (gradient, elapsed, N) -- the interior nodes of d(cost)/d(gamma) as a density,
        the wall time of the sensitivity alone, and the number of steps taken.
    """
    Nx = (n_el + 3,) * DIM
    dx = (LENGTH / n_el,) * DIM
    N = math.ceil(T / dt_stable)
    dt = T / N

    sim = ScalarWave(
        Nx,
        dx,
        N,
        dt,
        THREADS,
        precision=PRECISION,
        space_order=space_order,
        wavespeed=WAVESPEED,
        density=DENSITY,
    )

    # the indicator scales inertia and stiffness alike, so it is a density ratio: the
    # wave speed -- and with it the stable dt -- is the same inside the inclusion as
    # out, however violent the contrast
    x, y = grid_coords(Nx, dx, dtype=sim.dtype)
    r = cp.sqrt((x - 0.5 * LENGTH) ** 2 + (y - 0.5 * LENGTH) ** 2)
    if SMOOTHING:
        inside = 0.5 * (1.0 - cp.tanh((r - RADIUS) / SMOOTHING))
    else:
        inside = r < RADIUS
    true_indicator = (1.0 + (DENSITY0 / DENSITY - 1.0) * inside).astype(sim.dtype)

# --------------------------------------- source --------------------------------------
    signal = ricker(np.arange(N) * dt, AMPLITUDE, FREQUENCY)
    source = point_source(sim, [(0.0, 0.5 * LENGTH)], signal)
    # fixed physical points, so the misfit is the same functional on every level
    rows = np.arange(1, SENSOR_DIVISOR) * (n_el // SENSOR_DIVISOR) + 1
    sensors = cp.array([np.full(rows.size, 1), rows], dtype=cp.int32)

# -------------------------------------- measure --------------------------------------
    _, observed = simulate(sim, source, true_indicator, sensors=sensors)
    indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    objective = l2_misfit(observed)

    cp.cuda.Stream.null.synchronize()
    tic = time.time()
    if METHOD == "standard":
        _, grads, _, _ = sensitivity(sim, source, indicator, sensors, objective)
    else:
        _, grads, _, _ = superposition_sensitivity(
            sim, source, indicator, sensors, objective, scale=SUPERPOSITION_SCALE
        )
    cp.cuda.Stream.null.synchronize()
    toc = time.time()

    # chain rule: the module differentiates w.r.t. the two material fields, the
    # parametrization maps them back onto the indicator
    d_mass, d_stiff = sim.parametrization_jacobian()
    gradient = d_mass * grads["mass"] + d_stiff * grads["stiff"]
    # the gradients come back over the *padded* grid, so the interior is taken against
    # the logical Nx: slice(1, -1) would reach into the padding of the fastest axis
    interior = tuple(slice(1, n - 1) for n in Nx)
    # the nodal gradient is an integral over the node's cell and over a sum the
    # timestep never weighted, so dt / dx**DIM turns it into the density the levels
    # share -- without it the comparison measures the units, not the discretization
    return (
        gradient[interior].copy() * (dt / np.prod(dx)),
        toc - tic,
        N,
    )


# ------------------------------------- reference -------------------------------------
dt_reference = REFERENCE_SAFETY * stable_dt(
    (LENGTH / REFERENCE,) * DIM, WAVESPEED, REFERENCE_ORDER
)
truth, elapsed, N = gradient_of(REFERENCE_ORDER, REFERENCE, dt_reference)
print(
    f"reference: {REFERENCE + 1}^{DIM} nodes at order {REFERENCE_ORDER}, {N} steps, "
    f"{elapsed:.2f} s, {mempool.total_bytes() / 1e9:.2f} GB"
)
mempool.free_all_blocks()

# --------------------------------------- sweep ---------------------------------------
results = {}
for space_order in SPACE_ORDERS:
    gradient_of(space_order, min(LEVELS), T)  # this order's nvcc compiles, untimed
    dofs, errors, timings = [], [], []
    for n_el in LEVELS:
        dt_stable = SAFETY * stable_dt((LENGTH / n_el,) * DIM, WAVESPEED, space_order)
        gradient, elapsed, N = gradient_of(space_order, n_el, dt_stable)

        dofs.append((n_el + 1) ** DIM)
        errors.append(
            l2_error(gradient, truth[(slice(None, None, REFERENCE // n_el),) * DIM])
        )
        timings.append(elapsed)
        mempool.free_all_blocks()
    results[space_order] = (dofs, errors, timings)
    print(f"order {space_order:2d}  " + "  ".join(f"{error:.2e}" for error in errors))

# ----------------------------------- postprocessing ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for space_order, color in zip(SPACE_ORDERS, COLORS):
    dofs, errors, timings = results[space_order]
    axes[0].plot(dofs, errors, color=color, marker="o", label=f"order {space_order}")
    axes[1].plot(timings, errors, color=color, marker="o")

axes[0].set_xlabel("dofs")
axes[1].set_xlabel("wall time [s]")
axes[0].set_ylabel("relative L2 error of the sensitivity")
for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
axes[0].legend()
fig.tight_layout()
plt.show()

levels = np.array(LEVELS, dtype=float)
for space_order, (_, errors, _) in results.items():
    errors = np.array(errors)
    rates = np.log(errors[:-1] / errors[1:]) / np.log(levels[1:] / levels[:-1])
    print(
        f"order {space_order:2d}  rates  " + "  ".join(f"{rate:5.2f}" for rate in rates)
    )
