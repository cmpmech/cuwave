"""Box elastodynamics: P and S fronts from a point force in the homogeneous strip.

Every edge carries a pml.
"""

import math
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cuwave.boundary import Clamped, Pml, pad_for_sponge
from cuwave.elastic import ElasticWave
from cuwave.signals import ricker
from cuwave.utils import point_source
from cuwave.wave import simulate, stable_dt

sys.path.insert(0, str(Path.home() / "pdes" / "code"))
import fieldimage  # noqa: E402

# -------------------------------------- settings -------------------------------------
# discretization
PRECISION = "float32"
THREADS = (8, 32)
SPACE_ORDER = 2
RESOLUTION = 400  # nodes per unit length, the strip is 6 x 1
SAFETY = 0.9  # fraction of stable time step

# physics
DENSITY = 1.0
WAVESPEED_P = 1.0
POISSON = 0.25  # c_p / c_s = sqrt((2 - 2 nu) / (1 - 2 nu)) in plane strain
T = 2.4

# pml
THICKNESS = 0.1  # layer depth on every edge, clamped behind it
REFLECTION = 1e-6

# source
SOURCE = (3.0, 0.2)
DIRECTION = (1.0, 0.0)  # a horizontal point force
AMPLITUDE, FREQUENCY = 1.0, 6.0  # Ricker peak frequency

RECORDED = 0.806644  # largest |div u|, default settings

# --------------------------------------- setup ---------------------------------------
wavespeed_s = WAVESPEED_P * math.sqrt((1 - 2 * POISSON) / (2 - 2 * POISSON))
dx = (1.0 / RESOLUTION,) * 2
Nx, width, origin, plate = pad_for_sponge(
    (fieldimage.LENGTH * RESOLUTION + 3, RESOLUTION + 3), dx, THICKNESS
)
dt = SAFETY * stable_dt(dx, WAVESPEED_P, SPACE_ORDER)
N = math.ceil(T / dt)

pml = Pml(width, WAVESPEED_P, REFLECTION, frequency=0.5 * FREQUENCY)
sim = ElasticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    boundary=Clamped,
    pml=pml,
    density=DENSITY,
    wavespeed_p=WAVESPEED_P,
    wavespeed_s=wavespeed_s,
)
indicator = cp.ones(Nx, dtype=sim.dtype)

print(f"{wavespeed_s / (FREQUENCY * dx[0]):.0f} points per shear wavelength")

# --------------------------------------- source --------------------------------------
signal = ricker(np.arange(N) * dt, AMPLITUDE, FREQUENCY) * np.prod(dx)  # force times V
source = point_source(sim, [np.add(SOURCE, origin)], signal, direction=DIRECTION)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator)
cp.cuda.Stream.null.synchronize()
print(f"{N} steps, elapsed time {time.time() - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
# staggered u: div u lands on the nodes, the rotation on the cell centres, then averaged
ux, uy = u[0], u[1]
h = dx[0]
nodal = lambda inner: cp.pad(inner, 1, constant_values=cp.nan)
div = nodal((ux[1:-1, 1:-1] - ux[:-2, 1:-1] + uy[1:-1, 1:-1] - uy[1:-1, :-2]) / h)
centred = (uy[1:, :-1] - uy[:-1, :-1] - ux[:-1, 1:] + ux[:-1, :-1]) / h
rotation = nodal(
    (centred[1:, 1:] + centred[:-1, 1:] + centred[1:, :-1] + centred[:-1, :-1]) / 4
)
fields = {
    name: field[plate].get()
    for name, field in (("elastodynamics_div", div), ("elastodynamics_curl", rotation))
}

for name, field in fields.items():
    scale = float(np.nanpercentile(np.abs(field), 99.5))
    fieldimage.save(name, field, "pride", -scale, scale)
    print(f"{name}: limits +-{scale:.4g}")
divmax = float(np.nanmax(np.abs(fields["elastodynamics_div"])))
print(f"largest |div u| {divmax:.5g}")
assert abs(divmax / RECORDED - 1) < 1e-3, divmax
