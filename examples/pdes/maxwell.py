"""Box maxwell: the E_z pulse of a line current (TM in 2D) scattered by the holes.

The holes are perfect conductors, their nodes held at E_z = 0. cuwave's pml refuses a
held-node mask, so every edge carries a sponge instead.
"""

import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import cupy as cp
import numpy as np

from cuwave.boundary import pad_for_sponge, sponge
from cuwave.maxwell import ElectricWave
from cuwave.signals import ricker
from cuwave.utils import point_source
from cuwave.wave import grid_coords, simulate, stable_dt

sys.path.insert(0, str(Path.home() / "pdes" / "code"))
import fieldimage  # noqa: E402

# -------------------------------------- settings -------------------------------------
# discretization
PRECISION = "float32"
THREADS = (4, 128)
SPACE_ORDER = 2  # the held conductor nodes are exact images at order 2
RESOLUTION = 400  # nodes per unit length, the strip is 6 x 1
SAFETY = 0.99  # fraction of stable time step

# physics
PERMITTIVITY = 1.0  # vacuum around perfectly conducting holes, E_z = 0 inside
PERMEABILITY = 1.0
TIMES = (2.0, 5.0)  # t1 (top image) and t2 (bottom image)

# sponge
THICKNESS = 8.0  # layer thickness in wavelengths, on every edge
BETA = 0.02  # peak damping

# source
SOURCE = (0.9, 0.75)  # a line current along x3, clear of the holes
AMPLITUDE, FREQUENCY = 1.0, 10.0  # Ricker peak frequency
RECEIVER = (0.9, 0.55)  # where the trace is checked against the free-space solution
WINDOW = 0.55  # before the first echo off a hole reaches it

RECORDED = 0.012764  # largest |E_z| at t2, default settings

# --------------------------------------- setup ---------------------------------------
speed = 1.0 / math.sqrt(PERMITTIVITY * PERMEABILITY)
wavelength = speed / FREQUENCY
dx = (1.0 / RESOLUTION,) * 2
Nx, width, origin, plate = pad_for_sponge(
    (fieldimage.LENGTH * RESOLUTION + 3, RESOLUTION + 3), dx, THICKNESS * wavelength
)
dt = SAFETY * stable_dt(dx, speed, SPACE_ORDER)

# the holes are perfect conductors held at E_z = 0, placed in whole nodes
index = grid_coords(Nx, (1.0, 1.0))
conductor = fieldimage.holes(*((k - width) * h for k, h in zip(index, dx)))
sim = ElectricWave(
    Nx,
    dx,
    1,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    permittivity1=PERMITTIVITY,
    permittivity2=PERMITTIVITY,
    permeability=PERMEABILITY,
    dirichlet=conductor,
)
indicator = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
sim = replace(sim, damping=sponge(sim, indicator, width, BETA))
receiver = [[round((p + o) / h) + 1] for p, o, h in zip(RECEIVER, origin, dx)]
sensors = cp.array(receiver, dtype=cp.int32)

print(f"{wavelength / dx[0]:.0f} points per wavelength, sponge of {width} nodes")

# --------------------------------------- solve ---------------------------------------
fields = []
cp.cuda.Stream.null.synchronize()
tic = time.time()
for T in TIMES:
    sim = replace(sim, N=math.ceil(T / dt))
    signal = ricker(np.arange(sim.N) * dt, AMPLITUDE, FREQUENCY)
    source = point_source(sim, [np.add(SOURCE, origin)], signal)
    u, um = simulate(sim, source, indicator, sensors=sensors)
    fields.append(cp.where(conductor[plate], cp.nan, u[plate]).get())
    if T == TIMES[0]:
        trace = um[:, 0].get()
cp.cuda.Stream.null.synchronize()
print(f"{sim.N} steps, elapsed time {time.time() - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
# free-space Green's function convolved with the Ricker, sigma = r cosh(angle)
t = (np.arange(len(trace)) + 1) * dt
r = math.dist(RECEIVER, SOURCE) / speed
top = np.arccosh(np.maximum(t / r, 1.0))
angles = np.linspace(0.0, 1.0, 2001)[None, :] * top[:, None]
pulse = ricker(t[:, None] - r * np.cosh(angles), AMPLITUDE, FREQUENCY)
exact = np.trapezoid(pulse, angles, axis=1) * PERMEABILITY / (2 * np.pi)
early = t < WINDOW
error = np.linalg.norm((trace - exact)[early]) / np.linalg.norm(exact[early])
print(f"trace against the free-space solution: relative error {error:.3f}")

# one color range for both times, so the decay shows
scale = max(float(np.nanpercentile(np.abs(u), 99.5)) for u in fields)
for name, u in zip(("maxwell_ez_t1", "maxwell_ez_t2"), fields):
    fieldimage.save(name, u, "pride", -scale, scale)
print(f"limits +-{scale:.4g}")
ezmax = float(np.nanmax(np.abs(fields[-1])))
print(f"largest |E_z| at t2 {ezmax:.5g}")
assert abs(ezmax / RECORDED - 1) < 1e-3, ezmax
