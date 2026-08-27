import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.boundary import random_layer
from cuwave.signals import sineburst
from cuwave.wave import ScalarWave, Source, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
DIM = 2  # fixed
PRECISION = "float32"
THREADS = (4, 128)
SPACE_ORDER = 2  # a randomly rough layer does not suit the wide high-order flux
RESOLUTION = 500
SAFETY = 0.99  # fraction of the stable time step

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
T = 3.2  # the front has reflected off axis 1 and dissolved into the axis 0 layers

# layer, absorbing the low and high faces of axis 0 while axis 1 stays reflecting
WIDTH = 250  # 150  # 150  # 150  # layer thickness in nodes, about twelve wavelengths
CONTRAST = 0.5
CORRELATION = 6  # 6  # 6  # scatterer size in nodes
SEED = 0

# source
AMPLITUDE = 1e8
CYCLES = 5
FREQUENCY = 20  # bounded by WAVESPEED / (20.0 * min(dx))

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION + 2 * WIDTH, RESOLUTION)
dx = (LENGTH / (RESOLUTION - 3),) * DIM
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
indicator *= random_layer(
    sim, WIDTH, CONTRAST, faces=(0, 1), correlation=CORRELATION, rng=SEED
)

print(f"{WAVESPEED / (FREQUENCY * max(dx)):.0f} points per wavelength")

# --------------------------------------- helper --------------------------------------
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
# ghost nodes and the layer are both outside the physical domain
interior = tuple(slice(w + 1, n - 1 - w) for w, n in zip((WIDTH, 0), Nx))
u_np = u[interior].get()
scale = float(np.max(np.abs(u_np)))

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(u_np.T, cmap="seismic", vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
