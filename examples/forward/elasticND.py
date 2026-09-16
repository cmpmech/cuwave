import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.elastic import ElasticWave
from cuwave.signals import sineburst
from cuwave.utils import point_source
from cuwave.wave import simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
DIM = 2
PRECISION = "float32"
SPACE_ORDER = 4
SAFETY = 0.9  # fraction of stable time step
# per dimension
THREADS = {1: (128,), 2: (8, 32), 3: (2, 8, 32)}[DIM]
RESOLUTION = {1: 2000, 2: 400, 3: 140}[DIM]
FREQUENCY = {1: 20, 2: 8, 3: 5}[DIM]  # bounded by WAVESPEED_S / (10.0 * max(dx))

# physics
LENGTH = 1
WAVESPEED_P = 1.0
WAVESPEED_S = 0.5
DENSITY = 1
PLANE = "strain"  # "strain" or "stress" (for 2D)
T = 0.6

# source
AMPLITUDE = 1e4
CYCLES = 3

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION,) * DIM
dx = tuple(LENGTH / (n - 3) for n in Nx)
dt = SAFETY * stable_dt(dx, WAVESPEED_P, SPACE_ORDER)
N = math.ceil(T / dt)

sim = ElasticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    density=DENSITY,
    wavespeed_p=WAVESPEED_P,
    wavespeed_s=WAVESPEED_S,
    plane=PLANE,
)
indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

print(f"{WAVESPEED_S / (FREQUENCY * max(dx)):.0f} points per shear wavelength")

# --------------------------------------- source --------------------------------------
t_np = np.linspace(0, (N - 1) * dt, N)
signal = sineburst(t_np, AMPLITUDE, FREQUENCY, CYCLES)

# point force along the last axis, so both wave types are excited away from it
centre = [0.5 * LENGTH] * DIM
direction = [0.0] * (DIM - 1) + [1.0]
source = point_source(sim, [centre], signal, direction=direction)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
u_np = u[-1].get()  # component the force acts along
if DIM == 3:
    u_np = u_np[Nx[0] // 2]  # source plane cuts the sphere at its equator
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
