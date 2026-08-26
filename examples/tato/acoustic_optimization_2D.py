import math
import time
from collections.abc import Callable

import cupy as cp
import cupy.typing as cpt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cuwave.optimization import Adam
from cuwave.regularization import DensityFilter, Projection
from cuwave.sensitivity import sensitivity
from cuwave.signals import sineburst
from cuwave.wave import AcousticWave, Source, grid_coords, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 2  # the adjoint is the exact transpose only at order 2
PRECISION = "float32"  # "float32" | "float64"
RESOLUTION = (192, 192)
SAFETY = 0.7  # fraction of the stable time step

# physics
# air, solid
DENSITY1, DENSITY2 = 1.204, 2643.0
BULK_MODULUS1, BULK_MODULUS2 = 1.419e5, 6.87e8
FREQUENCY = 200.0
CYCLES = 3
AMPLITUDE = 1e3
POINTS_PER_WAVELENGTH = 10
T = 0.10

# geometry (metres)
LENGTHS = (9.0, 9.0)
SOURCE_Y = 4.5
TARGET_CENTER = (8.0, 4.5)
TARGET_SIZE = 2.0
DESIGN_CENTER = (4.5, 4.5)
DESIGN_SIZE = 4.0  # centered

# optimization
ITERATIONS = 40
LEARNING_RATE = 0.05
DESIGN_START = 0.5
MINIMIZE = True

# regularization
RMIN = 0.2
ETA = 0.5
BETA0 = 1.0
BETA_GROWTH = 2.0
BETA_STEP = 8  # iterations between beta doublings
BETA_MAX = 64.0

# --------------------------------------- setup ---------------------------------------
Nx = RESOLUTION
dx = tuple(LENGTHS[d] / (Nx[d] - 3) for d in range(len(Nx)))
to_index = lambda coord: tuple(int(round(coord[d] / dx[d])) + 1 for d in range(len(Nx)))

wavespeeds = np.sqrt(np.array([BULK_MODULUS1 / DENSITY1, BULK_MODULUS2 / DENSITY2]))
dt = SAFETY * stable_dt(dx, np.max(wavespeeds), SPACE_ORDER)
N = math.ceil(T / dt)

f_max = np.min(wavespeeds) / (POINTS_PER_WAVELENGTH * max(dx))
assert FREQUENCY <= f_max, (
    f"FREQUENCY {FREQUENCY:.0f} Hz exceeds the max resolvable {f_max:.0f} Hz; "
    f"lower FREQUENCY or raise RESOLUTION"
)

# the design enters through the indicator argument, not the constructor, so one sim
# serves every iteration
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
signal = sineburst(t, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
src_j = to_index((0.0, SOURCE_Y))[1]
source = Source(
    cp.array([[1], [src_j]], dtype=cp.int32),
    cp.asarray(signal[:, None], dtype=sim.dtype),
)

half = TARGET_SIZE / 2
lo_i, lo_j = to_index((TARGET_CENTER[0] - half, TARGET_CENTER[1] - half))
hi_i, hi_j = to_index((TARGET_CENTER[0] + half, TARGET_CENTER[1] + half))
box_i, box_j = cp.meshgrid(
    cp.arange(lo_i, hi_i + 1), cp.arange(lo_j, hi_j + 1), indexing="ij"
)
sensors = cp.stack([box_i.ravel(), box_j.ravel()]).astype(cp.int32)

x, y = grid_coords(Nx, dx, dtype=sim.dtype)
region = (cp.abs(x - DESIGN_CENTER[0]) <= DESIGN_SIZE / 2) & (
    cp.abs(y - DESIGN_CENTER[1]) <= DESIGN_SIZE / 2
)


# ------------------------------------ optimization -----------------------------------
def box_energy(sim: AcousticWave) -> Callable:
    """Objective factory: J = 1/2 int_box int_t p^2, the energy leaking into the box."""
    scale = float(np.prod(sim.dx)) * sim.dt

    def objective(traces):
        return 0.5 * scale * float(cp.sum(traces**2)), scale * traces

    return objective


objective = box_energy(sim)
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
    cost, grads, _, _ = sensitivity(sim, source, design, sensors, objective)

    # chain rule: material fields -> indicator -> projection -> filter -> variables
    d_mass, d_stiff = sim.parametrization_jacobian()
    gradient = d_mass * grads["mass"] + d_stiff * grads["stiff"]
    gradient = density_filter.grad(
        variables, projection.grad(filtered, gradient * region)
    )
    variables = cp.clip(optimizer.step(variables, sign * gradient * region), 0.0, 1.0)

    history.append(cost)
    print(
        f"iteration {iteration}: noise {cost:.4e}  "
        f"{10 * math.log10(cost / history[0]):+.2f} dB  beta {projection.beta:.0f}"
    )

projection.set(beta=beta_of(ITERATIONS))
design, _ = physical(variables)
wavefield, traces = simulate(sim, source, design, sensors=sensors)
history.append(objective(traces)[0])

cp.cuda.Stream.null.synchronize()
print(
    f"noise {history[0]:.4e} -> {history[-1]:.4e}  "
    f"({10 * math.log10(history[-1] / history[0]):+.2f} dB)  "
    f"{ITERATIONS:d} iterations of {N:d} steps: {time.time() - tic:.1f}s"
)

# ----------------------------------- postprocessing ----------------------------------
interior = tuple(slice(1, n - 1) for n in Nx)
final = design[interior].get()
wavefield = wavefield[interior].get()

scale = 0.5 * np.max(np.abs(wavefield)) + 1e-30
material = np.zeros((*final.shape, 4))
material[..., 3] = final

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].semilogy(history, "k")
axes[1].imshow(wavefield.T, origin="lower", cmap="seismic", vmin=-scale, vmax=scale)
axes[1].imshow(material.transpose(1, 0, 2), origin="lower")
axes[1].add_patch(
    Rectangle(
        (lo_i - 1.5, lo_j - 1.5),
        hi_i - lo_i + 1,
        hi_j - lo_j + 1,
        fill=False,
        edgecolor="k",
    )
)
axes[1].plot(0, src_j - 1, "ko", markersize=4)
axes[1].set_aspect("equal")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.tight_layout()
plt.show()
