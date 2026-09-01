import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.elastic import ElasticWave, stable_timestep
from cuwave.sensitivity import (
    l2_misfit,
    sensitivity,
    superposition_sensitivity,
)
from cuwave.signals import ricker
from cuwave.utils import Sensors, point_source
from cuwave.wave import grid_coords, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
PRECISION = "float32"  # "float32" | "float64"
SPACE_ORDER = 2  # a wide stencil over a strong inclusion costs stable time step
METHOD = "standard"  # "standard" | "superposition", the memory-efficient alternative
SUPERPOSITION_SCALE = 1e-2  # aim for a cancellation near 1e4 in float32
RESOLUTION = 240
SAFETY = 0.9  # fraction of the stable time step

# physics
LENGTH = 1.0
WAVESPEED_P = 1.0
WAVESPEED_S = 0.5
DENSITY = 1.0
DENSITY0 = 1e-4  # inside the inclusion
PLANE = "strain"
FREQUENCY = 12.0
T = 1.2

# geometry
RADIUS = 0.1
NUM_SENSORS = 24

# postprocessing
MARGIN = 0.05

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION, RESOLUTION)
dx = tuple(LENGTH / (n - 3) for n in Nx)
dt = SAFETY * stable_dt(dx, WAVESPEED_P, SPACE_ORDER)
N = math.ceil(T / dt)

sim = ElasticWave(
    Nx,
    dx,
    N,
    dt,
    (8, 32),
    precision=PRECISION,
    space_order=SPACE_ORDER,
    density=DENSITY,
    wavespeed_p=WAVESPEED_P,
    wavespeed_s=WAVESPEED_S,
    plane=PLANE,
)

# a density ratio, so both wave speeds are the same inside the inclusion as out
x, y = grid_coords(Nx, dx, dtype=sim.dtype)
hole = (x - LENGTH / 2) ** 2 + (y - LENGTH / 2) ** 2 < RADIUS**2
true_indicator = cp.where(hole, DENSITY0 / DENSITY, 1.0).astype(sim.dtype)

# the inclusion is what tightens the step at a wide stencil, so measure it rather than
# trust the wave speed alone
limit = stable_timestep(sim, true_indicator)
if dt > limit:
    raise ValueError(
        f"space order {SPACE_ORDER} over a density ratio of {DENSITY0 / DENSITY:g} "
        f"needs SAFETY at most {SAFETY * limit / dt:.2f}, not {SAFETY}"
    )

# --------------------------------------- source --------------------------------------
t = np.linspace(0, (N - 1) * dt, N)
signal = ricker(t, 1.0, FREQUENCY)

# a normal point force on the top edge, read by receivers measuring the same component
source = point_source(sim, [[LENGTH / 2, LENGTH]], signal, direction=[0.0, 1.0])
sensor_coords = np.stack(
    [
        np.linspace(0.05 * LENGTH, 0.95 * LENGTH, NUM_SENSORS),
        np.full(NUM_SENSORS, LENGTH),
    ],
    axis=1,
)
sensors = Sensors(sim, sensor_coords, direction=[0.0, 1.0])

# --------------------------------------- solve ---------------------------------------
# reference measurement through the true model
_, record = simulate(sim, source, true_indicator, sensors=sensors.nodes)
observed = sensors.traces(record)

# sensitivity of the misfit at the homogeneous starting model
indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

objective = sensors.objective(l2_misfit(observed))
cp.cuda.Stream.null.synchronize()
tic = time.time()
if METHOD == "standard":
    cost, grads, traces, info = sensitivity(
        sim, source, indicator, sensors.nodes, objective
    )
else:
    cost, grads, traces, info = superposition_sensitivity(
        sim, source, indicator, sensors.nodes, objective, scale=SUPERPOSITION_SCALE
    )
# past ~1e6 in float32 the gradient is mostly round-off; raise SUPERPOSITION_SCALE
note = f"\t cancellation {info['cancellation']:.1e}" if info else ""
cp.cuda.Stream.null.synchronize()
elapsed = time.time() - tic
print(f"{METHOD}: cost {cost:.4e}\t {N:d} steps: {elapsed:.2f}s{note}")

# chain rule from the two material fields back onto the indicator
d_mass, d_stiff = sim.parametrization_jacobian(indicator)
gradient = (d_mass * grads["mass"] + d_stiff * grads["stiff"]).get()

# ----------------------------------- postprocessing ----------------------------------
interior = tuple(slice(1, n - 1) for n in Nx)
gradient = gradient[interior]

margin = round(MARGIN / dx[0])
scale = 0.3 * np.max(np.abs(gradient[margin:-margin, margin:-margin])) + 1e-30

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(true_indicator[interior].get().T, origin="lower", cmap="binary_r")
axes[0].set_title("true density")
axes[1].imshow(gradient.T, origin="lower", cmap="Spectral_r", vmin=-scale, vmax=scale)
axes[1].set_title("sensitivity")
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()
