import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.evals import f1_score, l2_error, pr_auc
from cuwave.geometry import circle
from cuwave.optimization import Adam
from cuwave.postprocessing import markers, show
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.utils import (
    Sensors,
    interior_slice,
    measure,
    misfit,
    misfit_gradient,
    resample,
    shots,
    threshold,
)
from cuwave.wave import grid_coords, stable_dt

# -------------------------------------- settings -------------------------------------
# implementation
THREADS = (1, 16, 32)

# discretization
SPACE_ORDER = 4
SPACE_ORDER_OBS = 8  # measurement is simulated more accurately than it is inverted
PRECISION = "float32"
RESOLUTION = (96, 96, 96)
SAFETY = 0.9
GAMMA_MIN = 1e-3

# physics
T = 2.5  # traversals per length (z)
WAVESPEED, DENSITY = 1.0, 1.0  # material
AMPLITUDE, FREQUENCY, CYCLES = 1.0, 8.0, 2  # source
# transducer, a square grid on the top surface
SOURCES_PER_AXIS, SENSORS_PER_AXIS = 2, 32
SOURCE_SPAN, SENSOR_SPAN = (0.25, 0.75), (0.02, 0.98)  # absolute values
# geometry
LENGTHS = (1.0, 1.0, 1.0)
# defects, a cylinder along x
CYLINDER_CENTER = (0.5, 0.5, 0.45)
CYLINDER_RADIUS, CYLINDER_LENGTH = 0.1, 0.5
GAMMA_VOID = 1e-4

# optimization
ITERS, LR = 60, 1e-1

# evaluation
THRESHOLD = 0.5

# --------------------------------------- setup ---------------------------------------
Nx = RESOLUTION
dx = tuple(LENGTHS[d] / (Nx[d] - 3) for d in range(len(Nx)))
Lx, Ly, Lz = LENGTHS

N = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER))) + 1
N_obs = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER_OBS))) + 1
dt, dt_obs = T / (N - 1), T / (N_obs - 1)

print(
    f"{WAVESPEED / (FREQUENCY * max(dx)):.0f} points per wavelength, "
    f"{CYLINDER_RADIUS / max(dx):.1f} per cylinder radius"
)

scalar_wave = lambda N, dt, space_order: ScalarWave(
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
sim = scalar_wave(N, dt, SPACE_ORDER)
sim_obs = scalar_wave(N_obs, dt_obs, SPACE_ORDER_OBS)

# ------------------------------------ measurement ------------------------------------
burst = lambda t: sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)
t = np.linspace(0, T, N)
surface = lambda span, count: np.array(
    [(x, y, Lz) for x in np.linspace(*span, count) for y in np.linspace(*span, count)]
)
source_coords = surface(SOURCE_SPAN, SOURCES_PER_AXIS)
sensor_coords = surface(SENSOR_SPAN, SENSORS_PER_AXIS)
sources = shots(sim, source_coords, burst(t))
sensors = Sensors(sim, sensor_coords)

t_obs = np.linspace(0, T, N_obs)
sources_obs = shots(sim_obs, source_coords, burst(t_obs))
sensors_obs = Sensors(sim_obs, sensor_coords)

coords = grid_coords(Nx, dx, dtype=sim.dtype)
xc, yc, zc = CYLINDER_CENTER
cylinder = circle(coords[1:], (yc, zc), CYLINDER_RADIUS)
cylinder &= cp.abs(coords[0] - xc) < CYLINDER_LENGTH / 2
truth = cp.where(cylinder, GAMMA_VOID, 1.0).astype(sim.dtype)

cp.cuda.Stream.null.synchronize()
tic = time.time()
observed = [
    resample(record, dt_obs, dt, N)
    for record in measure(sim_obs, sources_obs, truth, sensors_obs)
]
cp.cuda.Stream.null.synchronize()
print(
    f"{len(sources)} shots of {N_obs} steps at space order {SPACE_ORDER_OBS} "
    f"recorded at {sensors.count} receivers: {time.time() - tic:.1f}s"
)

# ------------------------------------ optimization -----------------------------------
gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
optimizer = Adam(lr=LR)
history = []

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERS):
    cost, gradient = misfit_gradient(sim, sources, gamma, sensors, observed)
    gamma = cp.clip(optimizer.step(gamma, gradient), GAMMA_MIN, 1.0)

    history.append(cost)
    print(f"{iteration}/{ITERS}: normalized misfit {cost / history[0]:.4e}")
# final misfit
history.append(misfit(sim, sources, gamma, sensors, observed))
cp.cuda.Stream.null.synchronize()
print(f"{ITERS}/{ITERS}: normalized misfit {history[-1] / history[0]:.4e}")
print(f"elapsed time: {time.time() - tic:.1f}s")

# ------------------------------------- evaluation ------------------------------------
interior = interior_slice(sim)
reference, recovered = truth[interior], gamma[interior]

print(f"\naverage precision (pr auc): {pr_auc(recovered, reference):.3f}")
print(
    f"f1 at gamma < {THRESHOLD}: "
    f"{f1_score(recovered, reference, threshold=THRESHOLD):.3f}"
)
print(
    "relative L2 error: {:.3f} (thresholded: {:.3f})".format(
        l2_error(recovered, reference),
        l2_error(threshold(recovered, THRESHOLD, GAMMA_VOID, 1.0), reference),
    )
)

# ----------------------------------- postprocessing ----------------------------------
# interior index of the cylinder center, node 1 being the origin
center = [round(c / d) for c, d in zip(CYLINDER_CENTER, dx)]
sections = (
    lambda f: f[:, :, center[2]],  # x-y at the cylinder axis depth
    lambda f: f[:, center[1], :],  # x-z along the cylinder axis
    lambda f: f[center[0], :, :],  # y-z across the cylinder axis
)
surface_axes = (None, (0, 2), (1, 2))

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for row, field in zip(axes, (reference, recovered)):
    for ax, section, plane in zip(row, sections, surface_axes):
        show(ax, indicator=section(field), cmap="hot")
        if plane is not None:
            planar = lambda points: points[:, plane]
            spacing = tuple(dx[d] for d in plane)
            markers(ax, planar(sensor_coords), spacing, origin=1, nodes=3, color="k")
            markers(ax, planar(source_coords), spacing, origin=1, nodes=3, color="r")
        ax.set_aspect("equal")
        ax.axis("off")
fig.tight_layout()
plt.show()
