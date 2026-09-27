import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.geometry import circle
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.wave import (
    Source,
    domain_cells,
    domain_tiles,
    grid_coords,
    simulate,
    stable_dt,
)

# -------------------------------------- settings -------------------------------------
# discretization
DIM = 2
PRECISION = "float32"
THREADS = (4, 128)
SPACE_ORDER = 2  # masked problems: the graded wall is exact at order 2
RESOLUTION = 500
SAFETY = 0.99  # fraction of stable time step

# physics
WAVESPEED = 0.5
DENSITY = 1
T = 1.1

# geometry
RADIUS = 0.5  # the box is its bounding square

# source
AMPLITUDE, FREQUENCY, CYCLES = 1e8, 12, 5
SOURCE = (0.3, 0.6)

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION,) * DIM
dx = tuple(2 * RADIUS / (n - 3) for n in Nx)
dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)
N = math.ceil(T / dt)

# only the circle is stepped, its staircase a zero-flux wall
center = (RADIUS,) * DIM
domain = circle(grid_coords(Nx, dx), center, RADIUS)

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
    domain=domain,
)
indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

print(f"{WAVESPEED / (FREQUENCY * max(dx)):.0f} points per wavelength")
print(f"{int(domain.sum()):,} of {math.prod(Nx):,} nodes inside the circle")

# --------------------------------------- source --------------------------------------
t_np = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_np, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)

source_pos = cp.array([[round(x / d) + 1] for x, d in zip(SOURCE, dx)], dtype=cp.int32)
source = Source(source_pos, signal)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
interior = (slice(1, Nx[0] - 1), slice(1, Nx[1] - 1))
u_np = u.get()[interior]  # the ghost rows carry the box wall's mirror copy
scale = float(np.max(np.abs(u_np)))

tiles = domain_tiles(sim, domain_cells(sim)).get()
grid = [-(-n // t) for n, t in zip(sim.Nx_padded, THREADS)]
launched = np.zeros(grid)
launched[(tiles >> 10) & 1023, tiles & 1023] = 1
launched = np.repeat(np.repeat(launched, THREADS[0], 0), THREADS[1], 1)[interior]
print(f"{math.prod(grid) - len(tiles)} of {math.prod(grid)} {THREADS} tiles skipped")

# the true boundary in interior node coordinates, node i drawn over [i - 1, i]
boundary = lambda: plt.Circle(
    [c / d + 0.5 for c, d in zip(center, dx)],
    RADIUS / dx[0],
    fill=False,
    color="k",
    lw=0.5,
)

skipped = np.where(launched, np.nan, 1.0)

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(u_np.T, cmap="seismic", vmin=-scale, vmax=scale)
ax.pcolormesh(skipped.T, cmap="binary", vmin=0, vmax=2)
ax.add_patch(boundary())
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
