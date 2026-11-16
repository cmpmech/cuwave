"""Box elastic_helmholtz: the ground under a vibrating foundation, at one frequency.

cuwave has no frequency-domain solver, so the time-harmonic amplitude is the DFT of a
time-domain run at the drive frequency once it is steady. A pml replaces the dashpots
of the box; the tunnels are a soft inclusion (GAMMA_VOID) approximating empty cavities.
"""

import math
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cuwave.boundary import Clamped, Pml, Traction, pad_for_sponge
from cuwave.elastic import ElasticWave
from cuwave.utils import point_source
from cuwave.wave import grid_coords, simulate, stable_dt

sys.path.insert(0, str(Path.home() / "pdes" / "code"))
import fieldimage  # noqa: E402

# -------------------------------------- settings -------------------------------------
# discretization
PRECISION = "float32"
THREADS = (8, 32)
SPACE_ORDER = 2  # a void costs no timestep at order 2
RESOLUTION = 400  # nodes per unit length, the ground is 6 x 1
SAFETY = 0.9  # fraction of stable time step
SAMPLES = 4  # snapshots per period for the DFT, exact for a pure cosine

# physics
DENSITY = 1.0
WAVESPEED_P = 1.0
POISSON = 0.25  # c_p / c_s = sqrt((2 - 2 nu) / (1 - 2 nu)) in plane strain
GAMMA_VOID = 1e-3  # density and stiffness scale in the tunnels, traction-free cavities
T = 16.0  # time domain run to the steady state, the cavities ring down slowly

# pml
THICKNESS = 0.1  # layer depth on the left, right and bottom edges, clamped behind it
REFLECTION = 1e-6

# load
FOUNDATION = (2.9, 3.1)  # the strip of the top edge the foundation loads
TRACTION, FREQUENCY = 1.0, 3.0  # vertical traction amplitude, drive frequency
RAMP = 2  # periods over which the drive is switched on
PERIODS = 4  # the DFT spans the last PERIODS periods

RECORDED = 0.08040  # largest |u|, default settings

# --------------------------------------- setup ---------------------------------------
wavespeed_s = WAVESPEED_P * math.sqrt((1 - 2 * POISSON) / (2 - 2 * POISSON))
dx = (1.0 / RESOLUTION,) * 2
faces = (0, 1, 2)  # left, right, bottom; the top is the free surface
Nx, width, origin, ground = pad_for_sponge(
    (fieldimage.LENGTH * RESOLUTION + 3, RESOLUTION + 3), dx, THICKNESS, faces
)

# a whole number of steps a period, a snapshot every `stride` of them
stride = math.ceil(1 / (FREQUENCY * SAFETY * stable_dt(dx, WAVESPEED_P) * SAMPLES))
dt = 1 / (FREQUENCY * SAMPLES * stride)
N = SAMPLES * stride * math.ceil(T * FREQUENCY)

# the ground with the shared holes, in whole nodes so it is placed without round-off
index = grid_coords(Nx, (1.0, 1.0))
void = fieldimage.holes((index[0] - width) * dx[0], (index[1] - width) * dx[1])
pml = Pml(width, WAVESPEED_P, REFLECTION, frequency=0.5 * FREQUENCY, faces=faces)
sim = ElasticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    boundary=((Clamped, Clamped), (Clamped, Traction)),
    pml=pml,
    density=DENSITY,
    wavespeed_p=WAVESPEED_P,
    wavespeed_s=wavespeed_s,
)
indicator = cp.where(void, GAMMA_VOID, 1.0).astype(sim.dtype)

print(f"{wavespeed_s / (FREQUENCY * dx[0]):.0f} points per shear wavelength, {N} steps")

# --------------------------------------- source --------------------------------------
# point forces of traction times spacing along the strip, the cosine ramped up
t = np.arange(N) * dt
ramp = np.sin(0.5 * np.pi * np.minimum(t * FREQUENCY / RAMP, 1.0)) ** 2
signal = TRACTION * dx[0] * ramp * np.cos(2 * np.pi * FREQUENCY * t) * np.prod(dx)
strip = np.arange(FOUNDATION[0], FOUNDATION[1] + 0.5 * dx[0], dx[0])
top = np.stack([strip + origin[0], np.full_like(strip, 1.0 + origin[1])], axis=1)
source = point_source(sim, top, signal, direction=(0.0, -1.0))

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
_, snapshots = simulate(sim, source, indicator, record_every=stride)
cp.cuda.Stream.null.synchronize()
print(f"elapsed time {time.time() - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
# u = Re(u_hat exp(i omega t)) over the last PERIODS periods, snapshot k after k stride
last = snapshots[-PERIODS * SAMPLES :]
times = (np.arange(len(snapshots))[-PERIODS * SAMPLES :] * stride + 1) * dt
phase = np.exp(-2j * np.pi * FREQUENCY * times)
u_hat = 2 * np.tensordot(phase, last, axes=1) / len(last)

# u_1 half a node up x1, u_2 half a node up x2, the top row's u_2 only below it
ux = 0.5 * (u_hat[0, 1:-1] + u_hat[0, :-2])[:, 1:-1]
uy = 0.5 * (u_hat[1, :, 1:-1] + u_hat[1, :, :-2])[1:-1]
uy[:, -1] = u_hat[1, 1:-1, -3]
ux, uy = (np.pad(v, 1)[ground] for v in (ux, uy))
holes = void[ground].get()
amplitude = np.where(holes, np.nan, np.hypot(np.abs(ux), np.abs(uy)))
vertical = np.where(holes, np.nan, uy.real)

# the surface wave near the right end travels at about the Rayleigh speed
ratio = (wavespeed_s / WAVESPEED_P) ** 2
roots = np.roots([1, -8, 24 - 16 * ratio, -16 * (1 - ratio)])  # (c_R / c_s)**2
rayleigh = wavespeed_s * math.sqrt(min(r.real for r in roots if 0 < r.real < 1))
right = uy[round(5.3 / dx[0]) : round(5.8 / dx[0]), -1]
slope = np.polyfit(np.arange(right.size) * dx[0], np.unwrap(np.angle(right)), 1)[0]
print(f"surface wave {2 * np.pi * FREQUENCY / -slope:.4f}, Rayleigh {rayleigh:.4f}")

top_amplitude = float(np.nanpercentile(amplitude, 99.5))
fieldimage.save(
    "elastic_helmholtz_umag", amplitude, "pride_sequential", 0, top_amplitude
)
scale = float(np.nanpercentile(np.abs(vertical), 99.5))
fieldimage.save("elastic_helmholtz_u2", vertical, "pride", -scale, scale)
print(f"limits |u| 0 to {top_amplitude:.4g}, Re u_2 +-{scale:.4g}")
umax = float(np.nanmax(amplitude))
print(f"largest |u| {umax:.5g}")
assert abs(umax / RECORDED - 1) < 1e-3, umax
