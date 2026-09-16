import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.scalar import ScalarWave
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
SPACE_ORDER = 2  # adjoint is exact at order 2
PRECISION = "float32"
METHOD = "standard"  # "standard" or "superposition" (memory-efficient alternative)
SUPERPOSITION_SCALE = 1e2
RESOLUTION = 240
SAFETY = 0.99

# physics
LENGTH = 1.0
WAVESPEED = 1.0
DENSITY = 1.0
DENSITY0 = 1e-4
FREQUENCY = 15.0
T = 2.0

# geometry
RADIUS = 0.1
NUM_SENSORS = 24

# postprocessing
MARGIN = 0.05

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION, RESOLUTION)
dx = tuple(LENGTH / (n - 3) for n in Nx)
dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)
N = math.ceil(T / dt)

sim = ScalarWave(
    Nx,
    dx,
    N,
    dt,
    (4, 128),
    precision=PRECISION,
    space_order=SPACE_ORDER,
    wavespeed=WAVESPEED,
    density=DENSITY,
)

x, y = grid_coords(Nx, dx, dtype=sim.dtype)
hole = (x - LENGTH / 2) ** 2 + (y - LENGTH / 2) ** 2 < RADIUS**2
true_indicator = cp.where(hole, DENSITY0 / DENSITY, 1.0).astype(sim.dtype)

# --------------------------------------- source --------------------------------------
t = np.linspace(0, (N - 1) * dt, N)
signal = ricker(t, 1.0, FREQUENCY)

source = point_source(sim, [[0.0, LENGTH / 2]], signal)
sensor_coords = np.stack(
    [
        np.full(NUM_SENSORS, 0.0),
        np.linspace(dx[1], LENGTH - dx[1], NUM_SENSORS),
    ],
    axis=1,
)
sensors = Sensors(sim, sensor_coords)

# --------------------------------------- solve ---------------------------------------
# reference measurement through the true model
_, record = simulate(sim, source, true_indicator, sensors=sensors.nodes)
observed = sensors.traces(record)

# design field guess
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
note = f"\t cancellation {info['cancellation']:.1e}" if info else ""
cp.cuda.Stream.null.synchronize()
elapsed = time.time() - tic
print(f"{METHOD}: cost {cost:.4e}\t {N:d} steps: {elapsed:.2f}s{note}")

# chain rule
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
