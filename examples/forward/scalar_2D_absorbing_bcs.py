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
T = 2.0  # the coda has re-entered the domain, so it shows next to the reflecting arcs

# layer, opening the low and high faces of axis 0 while axis 1 stays reflecting
THICKNESS = 2.0  # layer thickness in dominant wavelengths
GRAIN = 0.5  # scatterer size in dominant wavelengths
BOUNDS = (0.2, 1.8)  # impedance range, since gamma leaves the wave speed at WAVESPEED
SEED = 0

# source
AMPLITUDE = 1e8
CYCLES = 5
FREQUENCY = 10  # bounded by WAVESPEED / (20.0 * min(dx))

# --------------------------------------- setup ---------------------------------------
wavelength = WAVESPEED / FREQUENCY
dx = (LENGTH / (RESOLUTION - 3),) * DIM
width = round(THICKNESS * wavelength / dx[0])
correlation = round(GRAIN * wavelength / dx[0])
Nx = (RESOLUTION + 2 * width, RESOLUTION)
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
indicator = random_layer(
    sim,
    cp.ones(sim.Nx_padded, dtype=sim.dtype),
    width,
    correlation,
    BOUNDS,
    faces=(0, 1),
    rng=SEED,
)

print(f"{wavelength / max(dx):.0f} points per wavelength")
print(f"layer {width} nodes of {correlation}-node grains")

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
interior = tuple(slice(w + 1, n - 1 - w) for w, n in zip((width, 0), Nx))
u_np = u[interior].get()
scale = float(np.max(np.abs(u_np)))

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(u_np.T, cmap="seismic", vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
