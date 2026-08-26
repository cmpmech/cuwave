import math
import time

import cupy as cp
import cupy.typing as cpt
import matplotlib.pyplot as plt
import numpy as np

from cuwave.evals import l2_error
from cuwave.geometry import circle
from cuwave.signals import sineburst
from cuwave.utils import point_source
from cuwave.wave import ScalarWave, grid_coords, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 2  # fixed
PRECISION = "float64"  # float32 floors the relative error near 1e-7
THREADS = (4, 128)

# discretization
SPACE_ORDERS = (2, 4, 6, 8)
# nine levels evenly spaced in log from 32 to 512, alternating the ratios 3/2 and 4/3.
# every one has to divide REFERENCE, so that the reference restricts onto it by strided
# slicing -- interpolating instead would add an O(dx**2) error to the rate measured here
LEVELS = (32, 48, 64, 96, 128, 192, 256, 384, 512)  # cells per axis
REFERENCE = 1536  # cells per axis: 3x the finest level, and a multiple of every one
REFERENCE_SAFETY = 0.25  # the reference refines in time as well as in space
SAFETY = 0.99  # fraction of the stable time step, which every run takes in full

# physics
LENGTH = 1.0
WAVESPEED = 1.0
DENSITY = 1.0
T = 4.0  # four traversals, so the wave has reflected off every wall

# source
AMPLITUDE = 1.0
CYCLES = 3
FREQUENCY = 4.0  # 8 points per wavelength on the coarsest level, 128 on the finest

# geometry
HETEROGENEOUS = False
RADIUS = 0.15
GAMMA_HOLE = 1e-3

# postprocessing
COLORS = ("k", "b", "g", "r")  # one per entry of SPACE_ORDERS

# --------------------------------------- setup ---------------------------------------
if any(REFERENCE % n_el for n_el in LEVELS):
    raise ValueError(f"every level must divide REFERENCE={REFERENCE}: {LEVELS}")
if len(COLORS) < len(SPACE_ORDERS):
    raise ValueError(f"{len(COLORS)} colors for {len(SPACE_ORDERS)} space orders")

order_max = max(SPACE_ORDERS)
interior = (slice(1, -1),) * DIM  # the returned field still carries its ghost ring

print(
    f"{WAVESPEED / (FREQUENCY * LENGTH / min(LEVELS)):.0f} points per wavelength "
    f"on the coarsest level"
)


# --------------------------------------- helper --------------------------------------
def solve(
    space_order: int, n_el: int, dt_stable: float
) -> tuple[cpt.NDArray, float, int]:
    """Solve to time `T` on `n_el` cells per axis, at the largest dt below `dt_stable`.

    Args:
        space_order: finite difference order of the run.
        n_el: cells per axis, so the grid holds `n_el + 3` nodes including ghosts.
        dt_stable: upper bound on the timestep, which `T` is then divided into.

    Returns:
        (u, elapsed, N) -- the interior nodes at `T` shaped (n_el + 1,) * DIM, the
        wall time of the time loop, and the number of steps taken.
    """
    Nx = (n_el + 3,) * DIM
    dx = (LENGTH / n_el,) * DIM
    # dt divides T exactly, so every run in the sweep lands on the same final time:
    # simulate returns the field after N steps, at N * dt rather than (N - 1) * dt
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

    # the burst ends at CYCLES / FREQUENCY, far short of T: the log singularity a 2D
    # point source carries lives only while it fires, so the final field is smooth
    signal = sineburst(np.arange(N) * dt, AMPLITUDE, FREQUENCY, CYCLES)
    source = point_source(sim, [(0.0, 0.5 * LENGTH)], signal)

    cp.cuda.Stream.null.synchronize()
    tic = time.time()
    u = simulate(sim, source, indicator)
    cp.cuda.Stream.null.synchronize()
    toc = time.time()
    # copied off the view simulate returns, which pins its whole two-field buffer
    return u[interior].copy(), toc - tic, N


# ------------------------------------- reference -------------------------------------
dt_reference = REFERENCE_SAFETY * stable_dt(
    (LENGTH / REFERENCE,) * DIM, WAVESPEED, order_max
)
truth, elapsed, N = solve(order_max, REFERENCE, dt_reference)
print(
    f"reference: {REFERENCE + 1}^{DIM} nodes at order {order_max}, "
    f"{N} steps, {elapsed:.2f} s"
)

# --------------------------------------- sweep ---------------------------------------
results = {}
for space_order in SPACE_ORDERS:
    solve(space_order, min(LEVELS), T)  # this order's nvcc compile, outside the timings
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
    # pairwise, not one fitted slope: where a curve floors -- order 2 does, on the
    # grid-scale content the point source radiates -- a slope averaged over the whole
    # sweep reports neither the rate before the floor nor the floor itself
    errors = np.array(errors)
    rates = np.log(errors[:-1] / errors[1:]) / np.log(levels[1:] / levels[:-1])
    print(
        f"order {space_order:2d}  rates  " + "  ".join(f"{rate:5.2f}" for rate in rates)
    )
