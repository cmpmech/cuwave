"""Shared problem definition of the analog recurrent network, imported by both drivers.

The grid, the sponge, the transducers and the classifier readout are fixed here rather
than in either driver, so a design trained by `analog_rnn_train` cannot be evaluated by
`analog_rnn_eval` on a grid it does not fit. See https://doi.org/10.1126/sciadv.aay6946.
"""

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
import numpy as np
import numpy.typing as npt

from cuwave.boundary import pad_for_sponge, sponge
from cuwave.sensitivity import reconstruction_nodes
from cuwave.utils import Sensors, point_source
from cuwave.wave import AcousticWave, Source, stable_dt

# -------------------------------------- settings -------------------------------------
DATASET = Path(__file__).parent / "minecraft_mobs.npz"
MATERIAL = Path(__file__).parent / "analog_rnn_material.npy"

# geometry: source on the left wall, three probes on the right wall (one per class)
LENGTHS = (10.0, 5.0)
SOURCE = [(0.5, 2.5)]
SENSOR = [(9.5, 1.25), (9.5, 2.5), (9.5, 3.75)]

# discretization
RESOLUTION = (600, 300)  # finer resolves more of the clip, at proportionally more steps
SPACE_ORDER = 2  # above 2 the adjoint is only consistent, and a binary design is rough
PRECISION = "float32"
THREADS = (4, 64)
SAFETY = 0.99  # fraction of the stable time step
T = 1.5  # matches the clip length the dataset is padded to

# physics
# air, foam: 13x the impedance of air, so the design transmits instead of mirroring
RHO1, RHO2 = 1.204, 30.0
KAPPA1, KAPPA2 = 1.419e5, 9.72e5  # foam bulk modulus, c2 ~ 180 m/s
AMPLITUDE = 1e3
POINTS_PER_WAVELENGTH = 10

# boundary: sponged on every face, so the probe energies are not degenerate
SPONGE_THICKNESS = 0.5  # metres, so it does not need rescaling with RESOLUTION
BETA = 0.05  # peak sponge damping d * dt / 2m

# --------------------------------------- setup ---------------------------------------
dx = tuple(LENGTHS[d] / (RESOLUTION[d] - 3) for d in range(2))
Nx, width, origin, region = pad_for_sponge(RESOLUTION, dx, SPONGE_THICKNESS)
x0, y0 = origin  # where the region of interest starts, the sponge sitting before it

wavespeeds = np.sqrt(np.array([KAPPA1 / RHO1, KAPPA2 / RHO2]))
dt = SAFETY * stable_dt(dx, np.max(wavespeeds), SPACE_ORDER)
N = math.ceil(T / dt)
f_max = np.min(wavespeeds) / (POINTS_PER_WAVELENGTH * max(dx))

sim = AcousticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    rho1=RHO1,
    rho2=RHO2,
    kappa1=KAPPA1,
    kappa2=KAPPA2,
)
# damping sets a compile flag, so it has to be in place before any kernel compiles
sim = replace(
    sim, damping=sponge(sim, cp.ones(sim.Nx_padded, dtype=sim.dtype), width, BETA)
)

# node index of a coordinate of the region of interest, the sponge offset added
to_node = lambda coord: tuple(
    int(round((origin[d] + coord[d]) / dx[d])) + 1 for d in range(2)
)

# probe m is the class-m readout (hostile 0, neutral 1, passive 2)
source_coords = [(x0 + x, y0 + y) for x, y in SOURCE]
sensors = Sensors(sim, [(x0 + x, y0 + y) for x, y in SENSOR])

strip, _ = reconstruction_nodes(sim)
footprint = (N + 2) * strip.shape[1] * np.dtype(sim.dtype).itemsize
bins = int(2.0 * f_max * T) // 2 + 1  # rfft bins of a clip that fit under f_max

# the grid cannot propagate what it cannot resolve, so the band decides what the medium hears
spectra = np.abs(np.fft.rfft(np.load(DATASET)["X"], axis=1)) ** 2
retained = spectra[:, :bins].sum(axis=1) / spectra.sum(axis=1)
if retained.min() < 0.1:
    raise ValueError(
        f"f_max={f_max:.0f} Hz keeps only {retained.min():.1%} of a clip: the source "
        f"would be the recording's noise floor, so shorten LENGTHS"
    )

print(f"{Nx[0]} x {Nx[1]} nodes, {N} steps, {f_max:.0f} Hz max resolvable")
print(f"source keeps {100 * retained.min():.0f}-{100 * retained.max():.0f}% of a clip")
print(f"adjoint strip {strip.shape[1]} nodes, {footprint / 1e6:.0f} MB")


# --------------------------------------- source --------------------------------------
def load_source(clip: npt.NDArray[np.float32]) -> Source:
    """Low-pass `clip` to the resolvable [0, `f_max`] and inject it at `SOURCE`."""
    # the same time base either side, so the cut is at f_max of the recording as well
    wave = np.fft.irfft(np.fft.rfft(clip)[:bins], N)
    return point_source(sim, source_coords, AMPLITUDE * wave / np.max(np.abs(wave)))


# --------------------------------------- readout -------------------------------------
def probabilities(traces: cpt.NDArray) -> cpt.NDArray:
    """Class scores p = y / sum(y) from the probe energies y_m = sum_t u_m**2."""
    y = cp.sum(traces**2, axis=0)
    return y / cp.sum(y)


def cross_entropy(label: int, penalty: float = 0.0) -> Callable:
    """Objective factory: J = -log p[`label`] - `penalty` * log(sum y), and dJ/du.

    Args:
        label: index of the probe reading out the true class.
        penalty: weight of the amplitude term, which rewards energy reaching the probes
            at all rather than only how it splits across them.

    Returns:
        `objective(traces)` returning (cost, dcost/dtraces) for the (N, num_sensors)
        record, its columns ordered by class.
    """

    def objective(traces):
        y = cp.sum(traces**2, axis=0)
        total = float(cp.sum(y))
        cost = -math.log(float(y[label]) / total) - penalty * math.log(total)
        # dJ/dy_m, carried onto the traces by dy_m/du_m = 2 u_m
        dy = cp.full(y.shape, (1.0 - penalty) / total, dtype=traces.dtype)
        dy[label] -= 1.0 / float(y[label])
        return cost, 2.0 * dy * traces

    return objective
