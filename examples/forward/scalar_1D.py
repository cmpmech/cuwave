import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.signals import sineburst
from cuwave.wave import ScalarWave, Source, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
DIM = 1  # fixed
PRECISION = "float32"
THREADS = (128,)
SPACE_ORDER = 4
RESOLUTION = 500
SAFETY = 0.99  # fraction of the stable time step

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
T = 1

# source
AMPLITUDE = 1e8
CYCLES = 5
FREQUENCY = 15  # bounded by WAVESPEED / (20.0 * min(dx))

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION,) * DIM
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


# --------------------------------------- helper --------------------------------------
t_np = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_np, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)

source_pos = cp.array([[1] for n in Nx], dtype=cp.int32)
source = Source(source_pos, signal)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator, record_every=None)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
u_np = u.get()
scale = float(np.max(np.abs(u_np)))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(u_np, "k")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
