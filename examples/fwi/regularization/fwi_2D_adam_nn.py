import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch

from cuwave.evals import f1_score, l2_error, pr_auc, precision, recall
from cuwave.geometry import stacked_circles
from cuwave.nn import Generator, nn_params
from cuwave.signals import sineburst
from cuwave.utils import Sensors, line, measure, misfit, misfit_gradient, shots
from cuwave.wave import ScalarWave, grid_coords, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 2
PRECISION = "float32"
RESOLUTION = (256, 256)
SAFETY = 0.99
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
# defects
NUM_VOIDS = 4
VOID_MAX_RADIUS = 0.05
VOID_GROWTH_RATE = 0.5
GAMMA_VOID = 1e-4
VOID_YRANGE, VOID_X = (0.1, 0.5), 0.5

# neural network
CHANNELS = [16, 16, 16, 8, 1]
KERNEL = 5
SEED = 0  # result should be independent of SEED -> check over multiple seeds
LEARNABLE_INPUT = False
OUTPUT_BIAS = 3.0

# optimization
ITERS, LR = 60, 3e-3

# --------------------------------------- setup ---------------------------------------
device = torch.device("cuda")
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

Nx = RESOLUTION
dx = tuple(LENGTHS[d] / (Nx[d] - 3) for d in range(len(Nx)))
Lx, Ly = LENGTHS

dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)
N = math.ceil(T / dt)

# about 12 points per wavelength is reasonable
print(
    f"{WAVESPEED / (FREQUENCY * max(dx)):.0f} points per wavelength, "
    f"{VOID_MAX_RADIUS * VOID_GROWTH_RATE ** (NUM_VOIDS - 1) / max(dx):.1f} per smallest radius"
)

sim = ScalarWave(
    Nx,
    dx,
    N,
    dt,
    (4, 64),
    precision=PRECISION,
    space_order=SPACE_ORDER,
    wavespeed=WAVESPEED,
    density=DENSITY,
)

# ------------------------------------ measurement ------------------------------------
t = np.linspace(0, (N - 1) * dt, N)
signal = sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)
surface = lambda count: line((ARRAY_SPAN[0], Ly), (ARRAY_SPAN[1], Ly), count)
source_coords = surface(NUM_SOURCES)
sources = shots(sim, source_coords, signal)
sensors = Sensors(sim, surface(NUM_SENSORS))

coords = grid_coords(Nx, dx, dtype=sim.dtype)
voids = stacked_circles(
    coords,
    NUM_VOIDS,
    VOID_MAX_RADIUS,
    (VOID_YRANGE[0] * Ly, VOID_YRANGE[1] * Ly),
    VOID_X * Lx,
    axis=1,
    ratio=VOID_GROWTH_RATE,
    order="ascending",
)
truth = cp.where(voids, GAMMA_VOID, 1.0).astype(sim.dtype)

cp.cuda.Stream.null.synchronize()
tic = time.time()
observed = measure(sim, sources, truth, sensors)
cp.cuda.Stream.null.synchronize()
print(
    f"{NUM_SOURCES} shots of {N} steps recorded at {NUM_SENSORS} receivers: "
    f"{time.time() - tic:.1f}s"
)

# ------------------------------------ optimization -----------------------------------
generator = Generator(
    CHANNELS,
    sim.Nx_padded,
    kernel=KERNEL,
    dim=len(Nx),
    output_bias=OUTPUT_BIAS,
    learnable=LEARNABLE_INPUT,
).to(device)
# cupy and torch share the pointer: no copies
to_cupy = lambda field: cp.asarray(field.detach())[0, 0]
indicator = lambda field: GAMMA_MIN + (1.0 - GAMMA_MIN) * to_cupy(field)
chain = lambda gradient: torch.as_tensor((1.0 - GAMMA_MIN) * gradient)[None, None]

optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
history = []

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERS):
    field = generator()
    cost, gradient = misfit_gradient(sim, sources, indicator(field), sensors, observed)
    # the step is taken in the weights: gradient is chained through the network
    optimizer.zero_grad()
    field.backward(chain(gradient))
    optimizer.step()

    history.append(cost)
    print(f"{iteration}/{ITERS}: normalized misfit {cost / history[0]:.4e}")
# final misfit
with torch.no_grad():
    gamma = indicator(generator()).copy()
history.append(misfit(sim, sources, gamma, sensors, observed))
cp.cuda.Stream.null.synchronize()
print(f"{ITERS}/{ITERS}: normalized misfit {history[-1] / history[0]:.4e}")
print(f"elapsed time: {time.time() - tic:.1f}s")

weights = nn_params(generator)
print(f"\n{weights} weights for {Nx[0] * Nx[1]} dofs\n")
# ------------------------------------- evaluation ------------------------------------
THRESHOLD = 0.5
interior = tuple(slice(1, n - 1) for n in Nx)
segmented = cp.where(gamma < THRESHOLD, GAMMA_VOID, 1.0).astype(sim.dtype)
reference, recovered = truth[interior], gamma[interior]
thresholded = segmented[interior]
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
    field[interior].get().T,
    origin="lower",
    extent=[0.0, Lx, 0.0, Ly],
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
    ax.plot(sensors.coordinates[:, 0], sensors.coordinates[:, 1], "ko", ms=5, **marker)
    ax.plot(source_coords[:, 0], source_coords[:, 1], "ro", ms=3, **marker)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
