import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.evals import f1_score, l2_error, pr_auc, precision, recall
from cuwave.geometry import stacked_circles
from cuwave.optimization import Lbfgs
from cuwave.regularization import Projection, continuation
from cuwave.signals import sineburst
from cuwave.utils import (
    Sensors,
    line,
    measure,
    misfit,
    misfit_gradient,
    resample,
    shots,
)
from cuwave.wave import ScalarWave, grid_coords, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 4
SPACE_ORDER_OBS = 8  # the measurement is simulated more accurately than it is inverted
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

# projection
SCHEME = "linear"
BETA, BETA_MAX, STAGES = 2.0, 4.0, 4
ETA = 0.5  # threshold the projection sharpens about

# optimization
ITERS, PAIRS = 30, 4  # secant pairs L-BFGS keeps

# evaluation
THRESHOLD = 0.5
FIRST_STEP = 0.4  # in x, which spans [0, 1]
ARMIJO = 1e-4  # sufficient decrease the line search asks for
BACKTRACK = 0.5  # factor a rejected trial step shrinks by
MIN_ALPHA = 1e-3  # give up below this fraction of the proposed step

# --------------------------------------- setup ---------------------------------------
Nx = RESOLUTION
dx = tuple(LENGTHS[d] / (Nx[d] - 3) for d in range(len(Nx)))
Lx, Ly = LENGTHS

# both grids end exactly on T, so the record resamples onto dt without extrapolating
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
sim_obs = scalar_wave(N_obs, dt_obs, SPACE_ORDER_OBS)

# ------------------------------------ measurement ------------------------------------
burst = lambda t: sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)
t = np.linspace(0, T, N)
surface = lambda count: line((ARRAY_SPAN[0], Ly), (ARRAY_SPAN[1], Ly), count)
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
    (VOID_YRANGE[0] * Ly, VOID_YRANGE[1] * Ly),
    VOID_X * Lx,
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
projection = Projection(BETA, ETA)
betas = continuation(SCHEME, ITERS, BETA, BETA_MAX, STAGES)
indicator = lambda x: GAMMA_MIN + (1.0 - GAMMA_MIN) * projection(x)
chain = lambda x, gradient: projection.grad(x, (1.0 - GAMMA_MIN) * gradient)

x = cp.ones(sim.Nx_padded, dtype=sim.dtype)
optimizer = Lbfgs(
    k=PAIRS,
    first_step=FIRST_STEP,
    armijo=ARMIJO,
    shrink=BACKTRACK,
    min_alpha=MIN_ALPHA,
)
history = []

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERS):
    projection.set(beta=betas[iteration])
    cost, gradient = misfit_gradient(sim, sources, indicator(x), sensors, observed)
    x, trial_cost, alpha, trials = optimizer.search(
        x,
        chain(x, gradient),
        cost,
        lambda z: misfit(sim, sources, indicator(z), sensors, observed),
        lambda z: cp.clip(z, 0.0, 1.0),
    )

    history.append(cost)
    print(
        f"{iteration}/{ITERS}: normalized misfit {cost / history[0]:.4e}  "
        f"sharpness {betas[iteration]:.2f}, step fraction {alpha:.3f} found in "
        f"{trials} function evals"
    )
# final misfit
gamma = indicator(x)
history.append(trial_cost)
cp.cuda.Stream.null.synchronize()
print(f"{ITERS}/{ITERS}: normalized misfit {history[-1] / history[0]:.4e}")
print(f"elapsed time: {time.time() - tic:.1f}s")

# ------------------------------------- evaluation ------------------------------------
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
    ax.plot(sensor_coords[:, 0], sensor_coords[:, 1], "ko", ms=5, **marker)
    ax.plot(source_coords[:, 0], source_coords[:, 1], "ro", ms=3, **marker)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
