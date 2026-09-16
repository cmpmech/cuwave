import math
import time
from collections.abc import Callable

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cuwave.scalar import AcousticWave
from cuwave.sensitivity import sensitivity, superposition_sensitivity
from cuwave.signals import sineburst
from cuwave.wave import Source, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 2  # adjoint is exact at order 2
PRECISION = "float32"
METHOD = "standard"  # "standard" or "superposition" (memory-efficient alternative)
SUPERPOSITION_SCALE = 1e4
RESOLUTION = (192, 192)
SAFETY = 0.7

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

# postprocessing
PERCENTILE = 95.0

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

# design field guess
indicator = cp.zeros(sim.Nx_padded, dtype=sim.dtype)

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


# --------------------------------------- solve ---------------------------------------
def box_energy(sim: AcousticWave) -> Callable:
    """Objective factory: J = 1/2 int_box int_t p^2, the energy leaking into the box."""
    scale = float(np.prod(sim.dx)) * sim.dt

    def objective(traces):
        return 0.5 * scale * float(cp.sum(traces**2)), scale * traces

    return objective


objective = box_energy(sim)
cp.cuda.Stream.null.synchronize()
tic = time.time()
if METHOD == "standard":
    cost, grads, traces, info = sensitivity(sim, source, indicator, sensors, objective)
else:
    cost, grads, traces, info = superposition_sensitivity(
        sim, source, indicator, sensors, objective, scale=SUPERPOSITION_SCALE
    )
note = f"\t cancellation {info['cancellation']:.1e}" if info else ""
cp.cuda.Stream.null.synchronize()
elapsed = time.time() - tic
print(
    f"{METHOD}: noise energy {cost:.4e}\t"
    f" peak pressure {float(cp.max(cp.abs(traces))):.3e}\t"
    f" {N:d} steps: {elapsed:.2f}s{note}"
)

# chain rule
d_mass, d_stiff = sim.parametrization_jacobian(indicator)
gradient = (d_mass * grads["mass"] + d_stiff * grads["stiff"]).get()

# ----------------------------------- postprocessing ----------------------------------
interior = tuple(slice(1, n - 1) for n in Nx)
gradient = gradient[interior]

scale = np.percentile(np.abs(gradient), PERCENTILE) + 1e-30

fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(gradient.T, origin="lower", cmap="Spectral_r", vmin=-scale, vmax=scale)
ax.add_patch(
    Rectangle(
        (lo_i - 1.5, lo_j - 1.5),
        hi_i - lo_i + 1,
        hi_j - lo_j + 1,
        fill=False,
        edgecolor="k",
    )
)
ax.plot(0, src_j - 1, "ko", markersize=4)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
plt.show()
