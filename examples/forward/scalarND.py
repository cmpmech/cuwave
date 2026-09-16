import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.wave import Source, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
DIM = 3
PRECISION = "float32"
SPACE_ORDER = 8
SAFETY = 0.99  # fraction of stable time step
# per dimension
THREADS = {1: (128,), 2: (4, 128), 3: (1, 16, 32)}[DIM]
RESOLUTION = {1: 2000, 2: 500, 3: 250}[DIM]
FREQUENCY = {1: 40, 2: 12, 3: 8}[DIM]  # bounded by WAVESPEED / (20.0 * max(dx))

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
T = 1

# source
AMPLITUDE = 1e8
CYCLES = 5

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

# --------------------------------------- source --------------------------------------
t_np = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_np, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)

source_pos = cp.array([[n // 2] for n in Nx], dtype=cp.int32)
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
if DIM == 3:
    u_np = u_np[Nx[0] // 2]  # source plane cuts sphere at its equator
scale = float(np.max(np.abs(u_np)))

if DIM == 1:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(u_np, "k")
else:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pcolormesh(u_np.T, cmap="seismic", vmin=-scale, vmax=scale)
    ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
