import math
import time

import cupy as cp
import cupy.typing as cpt
import matplotlib.pyplot as plt
import numpy as np

from cuwave.evals import l2_error
from cuwave.geometry import circle
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.utils import point_source
from cuwave.wave import grid_coords, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 2
PRECISION = "float64"
THREADS = (4, 128)

# discretization
SPACE_ORDERS = (2, 4, 6, 8)
LEVELS = (32, 48, 64, 96, 128, 192, 256, 384, 512)  # cells per axis
REFERENCE = 1536  # cells per axis (multiple of every entry in LEVELS)
REFERENCE_SAFETY = 0.25
REFERENCE_ORDER = 12
SAFETY = 0.99

# physics
LENGTH = 1.0
WAVESPEED = 1.0
DENSITY = 1.0
T = 4.0

# source
AMPLITUDE = 1.0
CYCLES = 3
FREQUENCY = 4.0

# geometry
HETEROGENEOUS = False
RADIUS = 0.15
GAMMA_HOLE = 1e-3

# postprocessing
COLORS = ("k", "b", "g", "r")

# --------------------------------------- setup ---------------------------------------
if any(REFERENCE % n_el for n_el in LEVELS):
    raise ValueError(f"every level must divide REFERENCE={REFERENCE}: {LEVELS}")
if len(COLORS) < len(SPACE_ORDERS):
    raise ValueError(f"{len(COLORS)} colors for {len(SPACE_ORDERS)} space orders")

interior = (slice(1, -1),) * DIM  # remove ghost cells

print(
    f"{WAVESPEED / (FREQUENCY * LENGTH / min(LEVELS)):.0f} points per wavelength "
    f"on the coarsest level"
)


# --------------------------------------- helper --------------------------------------
def solve(
    space_order: int, n_el: int, dt_stable: float
) -> tuple[cpt.NDArray, float, int]:
    """Solve to time `T` on `n_el` cells per axis, at the largest dt below `dt_stable`.

    Returns:
        (u, elapsed, N): the interior nodes at `T` shaped (n_el + 1,) * DIM, the
        wall time of the time loop, and the number of steps taken.
    """
    Nx = (n_el + 3,) * DIM
    dx = (LENGTH / n_el,) * DIM
    # dt divides T exactly, so every level lands on the same final time
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
    indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    if HETEROGENEOUS:
        coords = grid_coords(Nx, dx, dtype=sim.dtype)
        indicator[circle(coords, (0.5 * LENGTH,) * DIM, RADIUS)] = GAMMA_HOLE

    signal = sineburst(np.arange(N) * dt, AMPLITUDE, FREQUENCY, CYCLES)
    source = point_source(sim, [(0.0, 0.5 * LENGTH)], signal)

    cp.cuda.Stream.null.synchronize()
    tic = time.time()
    u = simulate(sim, source, indicator)
    cp.cuda.Stream.null.synchronize()
    toc = time.time()
    return u[interior].copy(), toc - tic, N


# ------------------------------------- reference -------------------------------------
dt_reference = REFERENCE_SAFETY * stable_dt(
    (LENGTH / REFERENCE,) * DIM, WAVESPEED, REFERENCE_ORDER
)
truth, elapsed, N = solve(REFERENCE_ORDER, REFERENCE, dt_reference)
print(
    f"reference: {REFERENCE + 1}^{DIM} nodes at order {REFERENCE_ORDER}, "
    f"{N} steps, {elapsed:.2f} s"
)

# --------------------------------------- sweep ---------------------------------------
results = {}
for space_order in SPACE_ORDERS:
    solve(space_order, min(LEVELS), T)  # keep nvcc compile outside timings
    dofs, errors, timings = [], [], []
    for n_el in LEVELS:
        dt_stable = SAFETY * stable_dt((LENGTH / n_el,) * DIM, WAVESPEED, space_order)
        u, elapsed, N = solve(space_order, n_el, dt_stable)

        dofs.append((n_el + 1) ** DIM)
        errors.append(l2_error(u, truth[(slice(None, None, REFERENCE // n_el),) * DIM]))
        timings.append(elapsed)
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
axes[0].set_ylabel(f"relative L2 error at t = {T}")
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
