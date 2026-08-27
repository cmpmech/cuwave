import math
import time

import cupy as cp
import cupy.typing as cpt
import matplotlib.pyplot as plt
import numpy as np

from cuwave.evals import non_discreteness
from cuwave.geometry import box, nodes
from cuwave.optimization import Adam
from cuwave.postprocessing import markers, outline, show
from cuwave.regularization import DensityFilter, Projection
from cuwave.signals import sineburst
from cuwave.utils import (
    energy,
    interior_slice,
    point_source,
    response,
    response_gradient,
    threshold,
)
from cuwave.wave import AcousticWave, grid_coords, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 4
PRECISION = "float32"
RESOLUTION = (192, 192)
SAFETY = 0.99

# physics
# air, solid
DENSITY1, DENSITY2 = 1.204, 2643.0
BULK_MODULUS1, BULK_MODULUS2 = 1.419e5, 6.87e8
AMPLITUDE, FREQUENCY, CYCLES = 1e3, 200.0, 3
POINTS_PER_WAVELENGTH = 12
T = 0.10

# geometry (metres)
LENGTHS = (9.0, 9.0)
SOURCE_X, SOURCE_Y = 0.0, 4.5
TARGET_CENTER = (8.0, 4.5)
TARGET_SIZE = 2.0
DESIGN_CENTER = (4.5, 4.5)
DESIGN_SIZE = 4.0  # centered

# optimization
ITERATIONS = 40
LEARNING_RATE = 0.05
DESIGN_START = 0.0  # range [0, 1]
MINIMIZE = True

# regularization
RMIN = 0.2
ETA = 0.5
BETA0, BETA_GROWTH, BETA_STEP, BETA_MAX = 1.0, 2.0, 8, 64.0

# evaluation
THRESHOLD = 0.5  # the projection maps onto [0, 1], so its midpoint is the cut

# --------------------------------------- setup ---------------------------------------
Nx = RESOLUTION
dx = tuple(LENGTHS[d] / (Nx[d] - 3) for d in range(len(Nx)))

wavespeeds = np.sqrt(np.array([BULK_MODULUS1 / DENSITY1, BULK_MODULUS2 / DENSITY2]))
dt = SAFETY * stable_dt(dx, np.max(wavespeeds), SPACE_ORDER)
N = math.ceil(T / dt)

f_max = np.min(wavespeeds) / (POINTS_PER_WAVELENGTH * max(dx))
assert FREQUENCY <= f_max, (
    f"FREQUENCY {FREQUENCY:.0f} Hz exceeds the max resolvable {f_max:.0f} Hz; "
    f"lower FREQUENCY or raise RESOLUTION"
)

# the design enters through the indicator, not the constructor, so one sim serves all
sim = AcousticWave(
    Nx,
    dx,
    N,
    dt,
    (4, 64),
    precision=PRECISION,
    space_order=SPACE_ORDER,
    rho1=DENSITY1,
    rho2=DENSITY2,
    kappa1=BULK_MODULUS1,
    kappa2=BULK_MODULUS2,
)

# --------------------------------------- source --------------------------------------
t = np.linspace(0, (N - 1) * dt, N)
source = point_source(
    sim, (SOURCE_X, SOURCE_Y), sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)
)

coords = grid_coords(Nx, dx, dtype=sim.dtype)
sensors = nodes(box(coords, TARGET_CENTER, (TARGET_SIZE, TARGET_SIZE)))
region = box(coords, DESIGN_CENTER, (DESIGN_SIZE, DESIGN_SIZE))
print(f"{sensors.shape[1]} target nodes over {int(region.sum())} design nodes")

# ------------------------------------ optimization -----------------------------------
objective = energy(sim)
density_filter = DensityFilter(RMIN / min(dx), sim.Nx_padded, dtype=sim.dtype)
projection = Projection(BETA0, ETA)
beta_of = lambda i: min(BETA0 * BETA_GROWTH ** (i // BETA_STEP), BETA_MAX)


def physical(variables: cpt.NDArray) -> tuple[cpt.NDArray, cpt.NDArray]:
    """Filter then project the design variables: (physical field, filtered field)."""
    filtered = density_filter(variables)
    return projection(filtered) * region, filtered


variables = cp.where(region, DESIGN_START, 0.0).astype(sim.dtype)
optimizer = Adam(lr=LEARNING_RATE)
sign = 1.0 if MINIMIZE else -1.0
history = []

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERATIONS):
    projection.set(beta=beta_of(iteration))
    design, filtered = physical(variables)
    cost, gradient = response_gradient(sim, source, design, sensors, objective)

    # chain rule: indicator -> projection -> filter -> variables
    gradient = density_filter.grad(
        variables, projection.grad(filtered, gradient * region)
    )
    variables = cp.clip(optimizer.step(variables, sign * gradient * region), 0.0, 1.0)

    history.append(cost)
    print(
        f"iteration {iteration}: noise {cost:.4e}  "
        f"{10 * math.log10(cost / history[0]):+.2f} dB  beta {projection.beta:.0f}"
    )
cp.cuda.Stream.null.synchronize()
print(f"{ITERATIONS:d} iterations of {N:d} steps: {time.time() - tic:.1f}s")

# ------------------------------------- evaluation ------------------------------------
projection.set(beta=beta_of(ITERATIONS))
design, _ = physical(variables)
final = threshold(design, THRESHOLD, dtype=sim.dtype)

cp.cuda.Stream.null.synchronize()
grey_cost, _ = response(sim, source, design, sensors, objective)
final_cost, wavefield = response(sim, source, final, sensors, objective)
history.append(grey_cost)
cp.cuda.Stream.null.synchronize()

# the thresholded design is the one that can be built, so it is the one that is reported
print(
    f"\nnoise {history[0]:.4e} -> {grey_cost:.4e} grey -> {final_cost:.4e} thresholded  "
    f"({10 * math.log10(final_cost / history[0]):+.2f} dB)"
)
print(
    f"\tdiscretization price {10 * math.log10(final_cost / grey_cost):+.2f} dB  "
    f"at non-discreteness {non_discreteness(design, region):.3f}"
)
print(f"\tsolid fraction {float(design[region].sum()) / int(region.sum()):.3f}")

# ----------------------------------- postprocessing ----------------------------------
interior = interior_slice(sim)
scale = 0.5 * float(cp.max(cp.abs(wavefield[interior]))) + 1e-30

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].semilogy(history, "k")
show(axes[1], field=wavefield[interior], indicator=final[interior], scale=scale)
outline(
    axes[1], sensors.min(axis=1).get(), sensors.max(axis=1).get(), origin=1, color="k"
)
markers(axes[1], (SOURCE_X, SOURCE_Y), dx=dx, origin=1, nodes=6, color="k")
axes[1].set_aspect("equal")
axes[1].axis("off")
fig.tight_layout()
plt.show()
