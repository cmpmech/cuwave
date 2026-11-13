import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

os.environ["CUWAVE_DEVICE"] = "cpu"  # read once, when cuwave is first imported

from cuwave.backend import ON_GPU, cp  # noqa: E402
from cuwave.scalar import ScalarWave  # noqa: E402
from cuwave.signals import sineburst  # noqa: E402
from cuwave.wave import Source, simulate, stable_dt  # noqa: E402

# -------------------------------------- settings -------------------------------------
# discretization
PRECISION = "float32"
SPACE_ORDER = 4
SAFETY = 0.99  # fraction of stable time step
THREADS = (4, 32)  # a block's threads run in turn on one CPU thread
RESOLUTION = 400
FREQUENCY = 12  # bounded by WAVESPEED / (20.0 * max(dx))

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
T = 1

# source
AMPLITUDE = 1e8
CYCLES = 5

# --------------------------------------- setup ---------------------------------------
assert not ON_GPU
Nx = (RESOLUTION,) * 2
dx = tuple(LENGTH / (n - 3) for n in Nx)
dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)
N = math.ceil(T / dt)

sim = ScalarWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    wavespeed=WAVESPEED,
    density=DENSITY,
)
indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

print(f"{WAVESPEED / (FREQUENCY * max(dx)):.0f} points per wavelength")
print(f"{os.environ.get('OMP_NUM_THREADS', 'all')} OpenMP threads")

# --------------------------------------- source --------------------------------------
t_np = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_np, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)

source_pos = cp.array([[n // 2] for n in Nx], dtype=cp.int32)
source = Source(source_pos, signal)

# --------------------------------------- solve ---------------------------------------
tic = time.time()
u = simulate(sim, source, indicator, record_every=None)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
u_np = cp.asnumpy(u)
scale = float(np.max(np.abs(u_np)))

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(u_np.T, cmap="seismic", vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
