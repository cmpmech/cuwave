import math
import time
from dataclasses import replace

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.boundary import pad_for_sponge, sponge
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.utils import interior_slice
from cuwave.wave import Source, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
DIM = 2
PRECISION = "float32"
THREADS = (4, 128)
SPACE_ORDER = 4
RESOLUTION = 500
SAFETY = 0.99  # fraction of stable time step

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
T = 5.0

# sponge
FACES = (0, 1)
THICKNESS = 4.0  # layer thickness in dominant wavelengths
BETA = 0.05  # peak damping

# source
AMPLITUDE, FREQUENCY, CYCLES = 1e8, 10, 5

# --------------------------------------- setup ---------------------------------------
wavelength = WAVESPEED / FREQUENCY
dx = (LENGTH / (RESOLUTION - 3),) * DIM
Nx, width, _, domain = pad_for_sponge(
    (RESOLUTION,) * DIM, dx, THICKNESS * wavelength, FACES
)
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
sim = replace(sim, damping=sponge(sim, indicator, width, BETA, faces=FACES))

print(f"{wavelength / max(dx):.0f} points per wavelength")
print(f"sponge of {width} nodes on {len(FACES)} of {2 * DIM} faces at beta {BETA}")

# --------------------------------------- helper --------------------------------------
t_np = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_np, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)

source_pos = cp.array([[n // 2] for n in Nx], dtype=cp.int32)
source = Source(source_pos, signal)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
full_np = u[interior_slice(sim)].get()
u_np = u[domain].get()
scale = float(np.max(np.abs(full_np)))

offset = tuple(s.start - 1 for s in domain)
axes = [np.arange(n + 1) + o for n, o in zip(u_np.shape, offset)]

fig, ax = plt.subplots(figsize=(5 * full_np.shape[0] / full_np.shape[1], 5))
shade = dict(cmap="seismic", vmin=-scale, vmax=scale)
ax.pcolormesh(full_np.T, alpha=0.5, **shade)
ax.pcolormesh(*axes, u_np.T, **shade)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
