import math
import time
from dataclasses import replace

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.boundary import pad_for_sponge, sponge
from cuwave.elastic import ElasticWave, stable_timestep
from cuwave.evals import f1_score, l2_error, pr_auc
from cuwave.geometry import stacked_circles
from cuwave.optimization import Adam
from cuwave.sensitivity import reconstruction_sensitivity
from cuwave.signals import sineburst
from cuwave.utils import (
    Sensors,
    line,
    measure,
    misfit,
    misfit_gradient,
    resample,
    shots,
    threshold,
)
from cuwave.wave import grid_coords, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
PRECISION = "float32"
RESOLUTION = (256, 256)  # of the region of interest, the sponge added outside it
SPACE_ORDER = 4
SPACE_ORDER_OBS = 6  # the measurement is simulated more accurately than it is inverted
SAFETY = 0.9
SAFETY_OBS = 0.45  # the measurement is stepped finer as well as wider
GAMMA_MIN = 1e-3

# physics
T = 2.5  # traversals per length (y) at the pressure wave speed
WAVESPEED_P, WAVESPEED_S, DENSITY = 1.0, 0.5, 1.0  # material
PLANE = "strain"
AMPLITUDE, FREQUENCY, CYCLES = 1.0, 10.0, 2  # source
# transducer, a normal force read by receivers measuring the same component
NUM_SOURCES, NUM_SENSORS = 4, 32
ARRAY_SPAN = (0.1, 0.9)  # absolute values
NORMAL = (0.0, 1.0)
# geometry
LENGTHS = (1.0, 1.0)
# boundary, opening the left and right edges while the transducer edge stays free
SPONGE_FACES = (0, 1)
THICKNESS = 2.0  # sponge thickness in dominant shear wavelengths
BETA = 0.05  # peak sponge damping, the flat optimum from two wavelengths up
# defects
NUM_VOIDS = 4
VOID_MAX_RADIUS = 0.05
VOID_GROWTH_RATE = 0.5
GAMMA_VOID = 1e-2  # a wide stencil goes unstable an order below this
VOID_YRANGE, VOID_X = (0.1, 0.5), 0.5

# optimization
ITERS, LR = 60, 1e-1

# evaluation
THRESHOLD = 0.5

# --------------------------------------- setup ---------------------------------------
dx = tuple(LENGTHS[d] / (RESOLUTION[d] - 3) for d in range(len(RESOLUTION)))
Lx, Ly = LENGTHS
Nx, pad, origin, region = pad_for_sponge(
    RESOLUTION, dx, THICKNESS * WAVESPEED_S / FREQUENCY, SPONGE_FACES
)
x0, y0 = origin  # where the region of interest starts, the sponge sitting before it

N = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED_P, SPACE_ORDER))) + 1
N_obs = math.ceil(T / (SAFETY_OBS * stable_dt(dx, WAVESPEED_P, SPACE_ORDER))) + 1
dt, dt_obs = T / (N - 1), T / (N_obs - 1)

# about 12 points per shear wavelength is reasonable
smallest_radius = VOID_MAX_RADIUS * VOID_GROWTH_RATE ** (NUM_VOIDS - 1)
print(
    f"{WAVESPEED_S / (FREQUENCY * max(dx)):.0f} points per shear wavelength, "
    f"{smallest_radius / max(dx):.1f} per smallest radius"
)

elastic_wave = lambda N, dt, space_order: ElasticWave(
    Nx,
    dx,
    N,
    dt,
    (8, 32),
    precision=PRECISION,
    space_order=space_order,
    density=DENSITY,
    wavespeed_p=WAVESPEED_P,
    wavespeed_s=WAVESPEED_S,
    plane=PLANE,
)
sim = elastic_wave(N, dt, SPACE_ORDER)
# one damping coefficient for both discretizations, so each sees its own beta
damping = sponge(
    sim, cp.ones(sim.Nx_padded, dtype=sim.dtype), pad, BETA, faces=SPONGE_FACES
)
sim = replace(sim, damping=damping)
sim_obs = replace(elastic_wave(N_obs, dt_obs, SPACE_ORDER_OBS), damping=damping)
print(f"sponge {pad} nodes per side at beta {BETA}, {Nx[0]} x {Nx[1]} nodes in total")

# ------------------------------------ measurement ------------------------------------
burst = lambda t: sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)
t = np.linspace(0, T, N)
surface = lambda count: line(
    (x0 + ARRAY_SPAN[0], y0 + Ly), (x0 + ARRAY_SPAN[1], y0 + Ly), count
)
source_coords, sensor_coords = surface(NUM_SOURCES), surface(NUM_SENSORS)
sources = shots(sim, source_coords, burst(t), direction=NORMAL)
sensors = Sensors(sim, sensor_coords, direction=NORMAL)

t_obs = np.linspace(0, T, N_obs)
sources_obs = shots(sim_obs, source_coords, burst(t_obs), direction=NORMAL)
sensors_obs = Sensors(sim_obs, sensor_coords, direction=NORMAL)

coords = grid_coords(Nx, dx, dtype=sim.dtype)
voids = stacked_circles(
    coords,
    NUM_VOIDS,
    VOID_MAX_RADIUS,
    (y0 + VOID_YRANGE[0] * Ly, y0 + VOID_YRANGE[1] * Ly),
    x0 + VOID_X * Lx,
    axis=1,
    ratio=VOID_GROWTH_RATE,
    order="ascending",
)
truth = cp.where(voids, GAMMA_VOID, 1.0).astype(sim.dtype)

# the void is what tightens the step at a wide stencil, so measure it against the truth
# rather than trust the wave speed alone
limit = stable_timestep(sim_obs, truth)
if dt_obs > limit:
    raise ValueError(
        f"space order {SPACE_ORDER_OBS} over a gamma of {GAMMA_VOID:g} needs "
        f"SAFETY_OBS at most {SAFETY_OBS * limit / dt_obs:.2f}, not {SAFETY_OBS}"
    )

cp.cuda.Stream.null.synchronize()
tic = time.time()
observed = [
    resample(record, dt_obs, dt, N)
    for record in measure(sim_obs, sources_obs, truth, sensors_obs)
]
cp.cuda.Stream.null.synchronize()
print(
    f"{NUM_SOURCES} shots of {N_obs} steps recorded at {NUM_SENSORS} receivers: "
    f"{time.time() - tic:.1f}s"
)

# ------------------------------------ optimization -----------------------------------
# the sponge is boundary and not design, so the optimizer never sees those nodes
design = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
design[region] = 1.0

gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
optimizer = Adam(lr=LR)
history = []

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERS):
    cost, gradient = misfit_gradient(
        sim, sources, gamma, sensors, observed, adjoint=reconstruction_sensitivity
    )
    gamma = cp.clip(optimizer.step(gamma, gradient * design), GAMMA_MIN, 1.0)

    history.append(cost)
    print(f"{iteration}/{ITERS}: normalized misfit {cost / history[0]:.4e}")
# final misfit
history.append(misfit(sim, sources, gamma, sensors, observed))
cp.cuda.Stream.null.synchronize()
print(f"{ITERS}/{ITERS}: normalized misfit {history[-1] / history[0]:.4e}")
print(f"elapsed time: {time.time() - tic:.1f}s")

# ------------------------------------- evaluation ------------------------------------
segmented = threshold(gamma, THRESHOLD, GAMMA_VOID, 1.0, sim.dtype)
reference, recovered = truth[region], gamma[region]
thresholded = segmented[region]
compare = lambda metric, **kwargs: (
    metric(recovered, reference, **kwargs),
    metric(thresholded, reference, **kwargs),
)

print("\neval: raw (thresholded)")
print("\taverage precision (pr auc): {:.3f} ({:.3f})".format(*compare(pr_auc)))
print(
    f"\tf1 at gamma < {THRESHOLD}: "
    f"{f1_score(recovered, reference, threshold=THRESHOLD):.3f}"
)
print("\trelative L2 error: {:.3f} ({:.3f})".format(*compare(l2_error)))

# ----------------------------------- postprocessing ----------------------------------
show_field = lambda ax, field: ax.imshow(
    field[region].get().T,
    origin="lower",
    extent=[x0, x0 + Lx, y0, y0 + Ly],
    cmap="hot",
    vmin=0,
    vmax=1.0,
)

fig, axes = plt.subplots(1, 4, figsize=(9, 3))
axes[0].semilogy(history, "k")
for ax, field, title in zip(
    axes[1:], (truth, gamma, segmented), ("truth", "reconstruction", "thresholded")
):
    show_field(ax, field)
    marker = dict(clip_on=False, zorder=3)
    ax.plot(sensor_coords[:, 0], sensor_coords[:, 1], "ko", ms=5, **marker)
    ax.plot(source_coords[:, 0], source_coords[:, 1], "ro", ms=3, **marker)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
