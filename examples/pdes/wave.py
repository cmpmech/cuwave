"""Box wave: a pulse from a point source in a room, its edges and holes sound hard."""

import itertools
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import cupy as cp
import numpy as np

from cuwave.scalar import ScalarWave
from cuwave.signals import ricker
from cuwave.utils import point_source
from cuwave.wave import grid_coords, simulate, stable_dt

sys.path.insert(0, str(Path.home() / "pdes" / "code"))
import fieldimage  # noqa: E402

# -------------------------------------- settings -------------------------------------
# discretization
PRECISION = "float32"
THREADS = (4, 128)
SPACE_ORDER = 2  # the masked wall is exact at order 2
RESOLUTION = 400  # nodes per unit length, the room is 6 x 1
SAFETY = 0.99  # fraction of stable time step

# physics
WAVESPEED = 1.0
DENSITY = 1.0
TIMES = (2.0, 5.0)  # t1 (top image) and t2 (bottom image)

# source
SOURCE = (0.15, 0.75)
AMPLITUDE, FREQUENCY = 1.0, 5.0  # Ricker peak frequency
RECEIVER = (0.35, 0.75)  # where the trace is checked against the free-space solution
WINDOW = 0.6  # before the first echo off a hole reaches it

RECORDED = 0.046499  # largest |u| at t2, default settings

# --------------------------------------- setup ---------------------------------------
Nx = (fieldimage.LENGTH * RESOLUTION + 3, RESOLUTION + 3)
dx = (1.0 / RESOLUTION,) * 2
dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)

# sound hard walls: the room's edges (Neumann, the default) and the holes' staircase
room = ~fieldimage.holes(*grid_coords(Nx, dx))
sim = ScalarWave(
    Nx,
    dx,
    1,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    wavespeed=WAVESPEED,
    density=DENSITY,
    domain=room,
)
indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
interior = (slice(1, Nx[0] - 1), slice(1, Nx[1] - 1))
sensors = cp.array([[round(x / h) + 1] for x, h in zip(RECEIVER, dx)], dtype=cp.int32)

print(f"{WAVESPEED / (FREQUENCY * dx[0]):.0f} points per wavelength")

# --------------------------------------- solve ---------------------------------------
fields = []
cp.cuda.Stream.null.synchronize()
tic = time.time()
for T in TIMES:
    sim = replace(sim, N=math.ceil(T / dt))
    signal = ricker(np.arange(sim.N) * dt, AMPLITUDE, FREQUENCY)
    source = point_source(sim, [SOURCE], signal)
    u, um = simulate(sim, source, indicator, sensors=sensors)
    fields.append(cp.where(room[interior], u[interior], cp.nan).get())
    if T == TIMES[0]:
        trace = um[:, 0].get()
cp.cuda.Stream.null.synchronize()
print(f"{sim.N} steps, elapsed time {time.time() - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
# free-space Green's function convolved with the Ricker, plus its mirrors in the left and top walls
t = (np.arange(len(trace)) + 1) * dt
theta = np.linspace(0.0, 1.0, 2001)
exact = np.zeros_like(t)
for sx, sy in itertools.product((SOURCE[0], -SOURCE[0]), (SOURCE[1], 2 - SOURCE[1])):
    r = math.hypot(RECEIVER[0] - sx, RECEIVER[1] - sy) / WAVESPEED
    top = np.arccosh(np.maximum(t / r, 1.0))
    angles = theta[None, :] * top[:, None]  # sigma = r cosh(angle) over [r, t]
    pulse = ricker(t[:, None] - r * np.cosh(angles), AMPLITUDE, FREQUENCY)
    exact += np.trapezoid(pulse, angles, axis=1) / (2 * np.pi * WAVESPEED**2 * DENSITY)
early = t < WINDOW
error = np.linalg.norm((trace - exact)[early]) / np.linalg.norm(exact[early])
print(f"trace against the free-space solution: relative error {error:.3f}")

scale = max(float(np.nanpercentile(np.abs(u), 99.5)) for u in fields)  # one color range for both times
print(f"color limits +-{scale:.4g}")
for name, u in zip(("wave_u_t1", "wave_u_t2"), fields):
    fieldimage.save(name, u, "pride", -scale, scale)
umax = float(np.nanmax(np.abs(fields[-1])))
print(f"largest |u| at t2 {umax:.5g}")
assert abs(umax / RECORDED - 1) < 1e-3, umax
