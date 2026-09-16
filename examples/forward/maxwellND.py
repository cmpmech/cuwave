import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.geometry import circles
from cuwave.maxwell import DielectricWave
from cuwave.signals import ricker
from cuwave.utils import point_source
from cuwave.wave import grid_coords, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
DIM = 2  # 2 or 3; a single in-plane component has no curl, so 1D is ElectricWave
PRECISION = "float32"
SPACE_ORDER = 4  # any even order; higher costs little since the curl is per axis
SAFETY = 0.9  # fraction of the stable time step

# per dimension
THREADS = {2: (8, 32), 3: (2, 8, 32)}[DIM]
RESOLUTION = {2: 400, 3: 140}[DIM]
FREQUENCY = {2: 12, 3: 5}[DIM]  # bounded by 1 / (24 INDEX max(dx))

# physics, nondimensional: the vacuum wavelength, speed and permittivity are all 1
LENGTH = 1
INDEX = 2.0
PERMITTIVITY1, PERMITTIVITY2 = 1.0, INDEX**2
PERMEABILITY = 1.0
T = 0.6

# geometry
SPHERE_CENTER, SPHERE_RADIUS = 0.62, 0.15
SOURCE_X = 0.3

# source
AMPLITUDE = 1.0
POLARIZATION = 1  # the axis the current runs along, which the source may not vary along

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION,) * DIM
dx = tuple(LENGTH / (n - 3) for n in Nx)
dt = SAFETY * stable_dt(dx, 1.0 / math.sqrt(PERMITTIVITY1 * PERMEABILITY), SPACE_ORDER)
N = math.ceil(T / dt)

sim = DielectricWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    permittivity1=PERMITTIVITY1,
    permittivity2=PERMITTIVITY2,
    permeability=PERMEABILITY,
)

coords = grid_coords(Nx, dx, dtype=sim.dtype)
indicator = circles(
    coords, [[SPHERE_CENTER] + [0.5 * LENGTH] * (DIM - 1)], SPHERE_RADIUS
).astype(sim.dtype)

print(f"{1.0 / (INDEX * FREQUENCY * max(dx)):.0f} points per wavelength in the sphere")

# --------------------------------------- source --------------------------------------
t_np = np.linspace(0, (N - 1) * dt, N)
signal = ricker(t_np, AMPLITUDE * dx[POLARIZATION], FREQUENCY)

# a line of currents spanning the polarization axis: uniform along it, so divergence free
line = np.arange(1, Nx[POLARIZATION] - 2) * dx[POLARIZATION]
origin = [SOURCE_X] + [0.5 * LENGTH] * (DIM - 1)
coordinates = []
for value in line:
    point = list(origin)
    point[POLARIZATION] = value
    coordinates.append(point)

direction = [0.0] * DIM
direction[POLARIZATION] = 1.0
source = point_source(sim, coordinates, signal, direction=direction)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
u_np = u[POLARIZATION].get()  # the component the current drives
if DIM == 3:
    u_np = u_np[:, :, Nx[2] // 2]  # the source plane, where the line looks like a sheet
scale = 0.2 * float(np.max(np.abs(u_np)))

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(u_np.T, cmap="seismic", vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
