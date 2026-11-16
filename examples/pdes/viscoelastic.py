"""Box viscoelastic: one pulse in an undamped and in a viscoelastic strip with holes.

The material is cuwave's generalized Zener body (`Relaxation`), whose memory variables
hold a constant Q over BAND, with Q_P = Q_S. A relaxation takes no pml, so every edge
carries a sponge; the holes are a soft inclusion (GAMMA_VOID) approximating cavities.
"""

import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import cupy as cp
import numpy as np

from cuwave.boundary import Clamped, pad_for_sponge, sponge
from cuwave.elastic import ElasticWave, Relaxation
from cuwave.signals import ricker
from cuwave.utils import point_source
from cuwave.wave import grid_coords, simulate, stable_dt

sys.path.insert(0, str(Path.home() / "pdes" / "code"))
import fieldimage  # noqa: E402

# -------------------------------------- settings -------------------------------------
# discretization
PRECISION = "float32"
THREADS = (8, 32)
SPACE_ORDER = 2  # a void costs no timestep at order 2
RESOLUTION = 400  # nodes per unit length, the strip is 6 x 1
SAFETY = 0.9  # fraction of stable time step, covering the unrelaxed speed

# physics
DENSITY = 1.0
WAVESPEED_P = 1.0  # phase speed at the source's peak frequency
POISSON = 0.25  # c_p / c_s = sqrt((2 - 2 nu) / (1 - 2 nu)) in plane strain
GAMMA_VOID = 1e-3  # density and stiffness scale in the holes, a traction-free void
QUALITY = 20.0  # Q of both moduli over the band, the bottom image
BAND = (1.0, 20.0)  # frequencies the generalized Maxwell body holds Q over
MECHANISMS = 3
T = 4.5

# sponge
THICKNESS = 6.0  # layer thickness in shear wavelengths, on every edge
BETA = 0.01  # peak damping

# source
SOURCE = (0.15, 0.75)
DIRECTION = (1.0, 0.0)  # a horizontal point force
AMPLITUDE, FREQUENCY = 1.0, 3.0  # Ricker peak frequency

RECORDED = 0.0035071  # largest |u_1| in the viscoelastic strip, default settings

# --------------------------------------- setup ---------------------------------------
wavespeed_s = WAVESPEED_P * math.sqrt((1 - 2 * POISSON) / (2 - 2 * POISSON))
dx = (1.0 / RESOLUTION,) * 2
Nx, width, origin, plate = pad_for_sponge(
    (fieldimage.LENGTH * RESOLUTION + 3, RESOLUTION + 3),
    dx,
    THICKNESS * wavespeed_s / FREQUENCY,
)
dt = SAFETY * stable_dt(dx, WAVESPEED_P, SPACE_ORDER)
N = math.ceil(T / dt)

# the strip with the shared holes, in whole nodes so it is placed without round-off
index = grid_coords(Nx, (1.0, 1.0))
void = fieldimage.holes(*((k - width) * h for k, h in zip(index, dx)))
sim = ElasticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    boundary=Clamped,
    density=DENSITY,
    wavespeed_p=WAVESPEED_P,
    wavespeed_s=wavespeed_s,
)
indicator = cp.where(void, GAMMA_VOID, 1.0).astype(sim.dtype)
sim = replace(sim, damping=sponge(sim, indicator, width, BETA))
relaxation = Relaxation(QUALITY, BAND, FREQUENCY, MECHANISMS)

print(f"{wavespeed_s / (FREQUENCY * dx[0]):.0f} points per shear wavelength")

# --------------------------------------- source --------------------------------------
signal = ricker(np.arange(N) * dt, AMPLITUDE, FREQUENCY) * np.prod(dx)  # force times V
source = point_source(sim, [np.add(SOURCE, origin)], signal, direction=DIRECTION)

# --------------------------------------- solve ---------------------------------------
fields = {}
cp.cuda.Stream.null.synchronize()
tic = time.time()
bodies = {"undamped": sim, "damped": replace(sim, relaxation=relaxation)}
for name, body in bodies.items():
    u = simulate(body, source, indicator)
    # u_1 sits half a node up x1, so the nodal value is the mean of its two neighbours
    ux = cp.pad(0.5 * (u[0, :-2, 1:-1] + u[0, 1:-1, 1:-1]), 1)
    fields[f"viscoelastic_u1_{name}"] = cp.where(void[plate], cp.nan, ux[plate]).get()
cp.cuda.Stream.null.synchronize()
print(f"2 x {N} steps, elapsed time {time.time() - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
# one scale for both, so the attenuation shows
scale = float(np.nanpercentile(np.abs(fields["viscoelastic_u1_undamped"]), 99.5))
for name, field in fields.items():
    fieldimage.save(name, field, "pride", -scale, scale)
print(f"limits +-{scale:.4g}")
umax = float(np.nanmax(np.abs(fields["viscoelastic_u1_damped"])))
print(f"largest |u_1| in the viscoelastic strip {umax:.5g}")
assert abs(umax / RECORDED - 1) < 1e-3, umax
