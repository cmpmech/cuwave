import math
import time
from dataclasses import replace

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from cuwave.boundary import pad_for_sponge, sponge
from cuwave.evals import f1_score, l2_error, pr_auc
from cuwave.geometry import stacked_circles
from cuwave.nn import Generator, nn_params
from cuwave.scalar import ScalarWave
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
SPACE_ORDER = 4
SPACE_ORDER_OBS = 8  # measurement is simulated more accurately than it is inverted
PRECISION = "float32"
RESOLUTION = (256, 256)  # of the region of interest, sponge added outside it
SAFETY = 0.9
GAMMA_MIN = 1e-3

# physics
T = 2.5  # traversals per length (y)
WAVESPEED, DENSITY = 1.0, 1.0  # material
AMPLITUDE, FREQUENCY, CYCLES = 1.0, 20.0, 2  # source
# transducer
NUM_SOURCES, NUM_SENSORS = 4, 32
ARRAY_SPAN = (0.1, 0.9)  # absolute values
# geometry
LENGTHS = (1.0, 1.0)
# boundary
SPONGE_FACES = (0, 1)
THICKNESS = 2.0  # sponge thickness in dominant wavelengths
BETA = 0.05  # peak sponge damping
# defects
NUM_VOIDS = 4
VOID_MAX_RADIUS = 0.05
VOID_GROWTH_RATE = 0.5
GAMMA_VOID = 1e-4
VOID_YRANGE, VOID_X = (0.1, 0.5), 0.5

# neural network
CHANNELS = [32, 32, 16, 8, 1]
KERNEL = 5
ACTIVATION = nn.Tanh  # over the GELU default, which stalls more often
SEED = 0  # result should be independent of SEED -> check over multiple seeds
LEARNABLE_INPUT = False
OUTPUT_BIAS = 3.0

# optimization
ITERS, LR = 60, 3e-3

# evaluation
THRESHOLD = 0.5

# --------------------------------------- setup ---------------------------------------
device = torch.device("cuda")
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

dx = tuple(LENGTHS[d] / (RESOLUTION[d] - 3) for d in range(len(RESOLUTION)))
Lx, Ly = LENGTHS
Nx, pad, origin, region = pad_for_sponge(
    RESOLUTION, dx, THICKNESS * WAVESPEED / FREQUENCY, SPONGE_FACES
)
x0, y0 = origin  # where the region of interest starts

N = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER))) + 1
N_obs = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER_OBS))) + 1
dt, dt_obs = T / (N - 1), T / (N_obs - 1)

# about 12 points per wavelength is reasonable
smallest_radius = VOID_MAX_RADIUS * VOID_GROWTH_RATE ** (NUM_VOIDS - 1)
print(
    f"{WAVESPEED / (FREQUENCY * max(dx)):.0f} points per wavelength, "
    f"{smallest_radius / max(dx):.1f} per smallest radius"
)

scalar_wave = lambda N, dt, space_order: ScalarWave(
    Nx,
    dx,
    N,
    dt,
    (4, 64),
    precision=PRECISION,
    space_order=space_order,
    wavespeed=WAVESPEED,
    density=DENSITY,
)
sim = scalar_wave(N, dt, SPACE_ORDER)
# one damping coefficient for both discretizations, so each sees its own beta
damping = sponge(
    sim, cp.ones(sim.Nx_padded, dtype=sim.dtype), pad, BETA, faces=SPONGE_FACES
)
sim = replace(sim, damping=damping)
sim_obs = replace(scalar_wave(N_obs, dt_obs, SPACE_ORDER_OBS), damping=damping)
print(f"sponge {pad} nodes per side at beta {BETA}, {Nx[0]} x {Nx[1]} nodes in total")

# ------------------------------------ measurement ------------------------------------
burst = lambda t: sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)
t = np.linspace(0, T, N)
surface = lambda count: line(
    (x0 + ARRAY_SPAN[0], y0 + Ly), (x0 + ARRAY_SPAN[1], y0 + Ly), count
)
source_coords, sensor_coords = surface(NUM_SOURCES), surface(NUM_SENSORS)
sources = shots(sim, source_coords, burst(t))
sensors = Sensors(sim, sensor_coords)

t_obs = np.linspace(0, T, N_obs)
sources_obs = shots(sim_obs, source_coords, burst(t_obs))
sensors_obs = Sensors(sim_obs, sensor_coords)

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

cp.cuda.Stream.null.synchronize()
tic = time.time()
observed = [
    resample(record, dt_obs, dt, N)
    for record in measure(sim_obs, sources_obs, truth, sensors_obs)
]
cp.cuda.Stream.null.synchronize()
print(
    f"{NUM_SOURCES} shots of {N_obs} steps at space order {SPACE_ORDER_OBS} "
    f"recorded at {NUM_SENSORS} receivers: {time.time() - tic:.1f}s"
)

# ------------------------------------ optimization -----------------------------------
# the design block is the region plus its ghost ring
block = tuple(slice(s.start - 1, s.stop + 1) for s in region)
generator = Generator(
    CHANNELS,
    RESOLUTION,
    kernel=KERNEL,
    activation=ACTIVATION,
    dim=len(RESOLUTION),
    output_bias=OUTPUT_BIAS,
    learnable=LEARNABLE_INPUT,
).to(device)
# cupy and torch share the pointer: no copies
to_cupy = lambda field: cp.asarray(field.detach())[0, 0]
chain = lambda gradient: torch.as_tensor((1.0 - GAMMA_MIN) * gradient)[None, None]

# sponge is boundary and not design, so those nodes keep the background
gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
history = []

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERS):
    field = generator()
    gamma[block] = GAMMA_MIN + (1.0 - GAMMA_MIN) * to_cupy(field)
    cost, gradient = misfit_gradient(
        sim, sources, gamma, sensors, observed, adjoint=reconstruction_sensitivity
    )
    # the step is taken in the weights, so the gradient chains through the network
    optimizer.zero_grad()
    field.backward(chain(gradient[block]))
    optimizer.step()

    history.append(cost)
    print(f"{iteration}/{ITERS}: normalized misfit {cost / history[0]:.4e}")
# final misfit
with torch.no_grad():
    gamma[block] = GAMMA_MIN + (1.0 - GAMMA_MIN) * to_cupy(generator())
history.append(misfit(sim, sources, gamma, sensors, observed))
cp.cuda.Stream.null.synchronize()
print(f"{ITERS}/{ITERS}: normalized misfit {history[-1] / history[0]:.4e}")
print(f"elapsed time: {time.time() - tic:.1f}s")

weights = nn_params(generator)
print(f"\n{weights} weights for {RESOLUTION[0] * RESOLUTION[1]} dofs\n")
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
