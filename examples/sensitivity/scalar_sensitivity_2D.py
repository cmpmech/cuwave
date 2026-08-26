import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.sensitivity import (
    l2_misfit,
    sensitivity,
    superposition_sensitivity,
)
from cuwave.signals import ricker
from cuwave.wave import (
    ScalarWave,
    Source,
    grid_coords,
    simulate,
    stable_dt,
)

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 2  # the adjoint is the exact transpose only at order 2
PRECISION = "float32"  # "float32" | "float64"
# "standard" keeps the forward history and is the exact discrete gradient;
# "superposition" reconstructs it by time reversal -- 3 fields instead of N + 2, at
# the price of a consistent-not-exact gradient and a scale to pick
METHOD = "standard"  # "standard" | "superposition"
SUPERPOSITION_SCALE = 1e2  # aim for a cancellation near 1e4 in float32
RESOLUTION = 240
SAFETY = 0.99  # fraction of the stable time step

# physics
LENGTH = 1.0
WAVESPEED = 1.0
DENSITY = 1.0
DENSITY0 = 1e-4  # inside the inclusion
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

# the indicator scales inertia and stiffness alike, so it is a density ratio: the
# wave speed -- and with it the stable dt -- is the same inside the inclusion as out,
# however violent the contrast
x, y = grid_coords(Nx, dx, dtype=sim.dtype)
hole = (x - LENGTH / 2) ** 2 + (y - LENGTH / 2) ** 2 < RADIUS**2
true_indicator = cp.where(hole, DENSITY0 / DENSITY, 1.0).astype(sim.dtype)

# --------------------------------------- source --------------------------------------
t = np.linspace(0, (N - 1) * dt, N)
signal = ricker(t, 1.0, FREQUENCY) / np.prod(dx)
source = Source(
    cp.array([[1], [Nx[1] // 2]], dtype=cp.int32),
    cp.asarray(signal[:, None], dtype=sim.dtype),
)
sensors = cp.array(
    [np.full(NUM_SENSORS, 1), np.linspace(2, Nx[1] - 3, NUM_SENSORS).astype(int)],
    dtype=cp.int32,
)

# --------------------------------------- solve ---------------------------------------
# reference measurement through the true model
_, observed = simulate(sim, source, true_indicator, sensors=sensors)

# sensitivity of the misfit at the homogeneous starting model
indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

objective = l2_misfit(observed)
cp.cuda.Stream.null.synchronize()
tic = time.time()
if METHOD == "standard":
    cost, grads, traces, info = sensitivity(sim, source, indicator, sensors, objective)
else:
    cost, grads, traces, info = superposition_sensitivity(
        sim, source, indicator, sensors, objective, scale=SUPERPOSITION_SCALE
    )
# how much of the mantissa the B(w, w) - B(u, u) subtraction ate: past ~1e6 in float32
# the gradient is mostly round-off and SUPERPOSITION_SCALE wants raising
note = f"\t cancellation {info['cancellation']:.1e}" if info else ""
cp.cuda.Stream.null.synchronize()
elapsed = time.time() - tic
print(f"{METHOD}: cost {cost:.4e}\t {N:d} steps: {elapsed:.2f}s{note}")

# chain rule: the module differentiates w.r.t. the two material fields, the
# parametrization maps them back onto the indicator
d_mass, d_stiff = sim.parametrization_jacobian()
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
