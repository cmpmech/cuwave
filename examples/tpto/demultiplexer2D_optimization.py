import math
import time
from dataclasses import replace

import cupy as cp
import cupy.typing as cpt
import matplotlib.pyplot as plt
import numpy as np

from cuwave.boundary import pad_for_sponge, sponge
from cuwave.evals import non_discreteness
from cuwave.geometry import box, nodes
from cuwave.maxwell import ElectricWave
from cuwave.optimization import Adam
from cuwave.postprocessing import outline, show
from cuwave.regularization import DensityFilter, Projection
from cuwave.sensitivity import reconstruction_sensitivity
from cuwave.signals import ricker
from cuwave.utils import intensity, point_source, response_gradient, threshold
from cuwave.wave import grid_coords, simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 4  # the design enters the inertia, whose gradient carries no stencil
PRECISION = "float32"
POINTS_PER_WAVELENGTH = 20  # in silicon, the short wavelength of the two
SAFETY = 0.95

# physics, nondimensional in the long wavelength: lambda_2 = c = eps_air = mu = 1
INDEX_SI = 3.48  # dispersion and loss are not modelled, so one index serves both bands
PERMITTIVITY1, PERMITTIVITY2 = 1.0, INDEX_SI**2
PERMEABILITY = 1.0
FREQUENCIES = (1.1923, 1.0)  # 1300 nm and 1550 nm, one broadband run scoring both
AMPLITUDE = 1.0
T = 60.0  # bin spacing 1/T well under the 0.19 the two frequencies are apart

# geometry (long wavelengths), from table 4 of Christiansen & Sigmund 2021 over 1550 nm
WIDTH, HEIGHT = 3.871, 1.935
DESIGN_SIZE = 1.290  # square, centred, the waveguides meeting its two vertical sides
GUIDE1, GUIDE2 = 0.193, 0.230  # heights, each sized for the wavelength it carries
OFFSET1, OFFSET2 = 0.330, -0.312  # output centres, above and below the input axis
SOURCE_X, PORT_INSET = 0.25, 0.20  # from the left wall, and from the right one

# boundary
THICKNESS = 3.0  # sponge thickness in long wavelengths, thick since guides run into it
SPONGE_DECAY = 4.3  # decay per wavelength travelled, so the layer follows the timestep

# optimization
ITERATIONS = 100
LEARNING_RATE = 0.05
DESIGN_START = 0.5  # range [0, 1]

# regularization
RMIN = 0.05  # filter radius in long wavelengths
ETA = 0.5
BETA0, BETA_GROWTH, BETA_STEP, BETA_MAX = 1.0, 2.0, 15, 64.0

# evaluation
THRESHOLD = 0.5  # the projection maps onto [0, 1], so its midpoint is the cut

# --------------------------------------- setup ---------------------------------------
dx = (1.0 / (INDEX_SI * POINTS_PER_WAVELENGTH),) * 2
resolution = tuple(int(length / dx[d]) + 3 for d, length in enumerate((WIDTH, HEIGHT)))
Nx, pad, origin, domain = pad_for_sponge(resolution, dx, THICKNESS)

dt = SAFETY * stable_dt(dx, 1.0, SPACE_ORDER)  # vacuum is the fast phase, not silicon
N = math.ceil(T / dt)
shift = lambda point: tuple(o + c for o, c in zip(origin, point))

sim = ElectricWave(
    Nx,
    dx,
    N,
    dt,
    (4, 64),
    precision=PRECISION,
    space_order=SPACE_ORDER,
    permittivity1=PERMITTIVITY1,
    permittivity2=PERMITTIVITY2,
    permeability=PERMEABILITY,
)
air = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
sim = replace(sim, damping=sponge(sim, air, pad, SPONGE_DECAY * dt))
print(
    f"{Nx[0]} x {Nx[1]} nodes, {pad} sponge, {N} steps of {dt:.4g}, "
    f"spectral bin {1.0 / (N * dt):.4f} against a separation of "
    f"{abs(FREQUENCIES[0] - FREQUENCIES[1]):.3f}"
)

coords = grid_coords(Nx, dx, dtype=sim.dtype)
axis = shift((0.0, 0.5 * HEIGHT))[1]
left = shift((0.5 * (WIDTH - DESIGN_SIZE), 0.0))[0]
right = shift((0.5 * (WIDTH + DESIGN_SIZE), 0.0))[0]


def guide(offset: float, height: float, inlet: bool) -> cpt.NDArray:
    """Silicon strip of `height` centred `offset` off the axis, running out through the layer."""
    core = cp.abs(coords[1] - (axis + offset)) <= 0.5 * height
    return core & (coords[0] <= left if inlet else coords[0] >= right)


def crossing(offset: float, height: float) -> cpt.NDArray[cp.int32]:
    """The nodes of a port: one guide cross-section a little inside the right wall."""
    at = shift((WIDTH - PORT_INSET, 0.0))[0]
    return nodes(
        (cp.abs(coords[1] - (axis + offset)) <= 0.5 * height)
        & (cp.abs(coords[0] - at) <= 0.5 * dx[0])
    )


waveguides = (
    guide(0.0, GUIDE1, True)
    | guide(OFFSET1, GUIDE1, False)
    | guide(OFFSET2, GUIDE2, False)
)
region = box(coords, shift((0.5 * WIDTH, 0.5 * HEIGHT)), (DESIGN_SIZE, DESIGN_SIZE))
ports = [crossing(OFFSET1, GUIDE1), crossing(OFFSET2, GUIDE2)]
sensors = cp.concatenate(ports, axis=1)
print(
    f"{int(region.sum())} design nodes, ports of {ports[0].shape[1]} and "
    f"{ports[1].shape[1]} nodes"
)

# each wavelength is scored on its own port, which is the whole demultiplexing objective
weights = cp.zeros((len(FREQUENCIES), sensors.shape[1]), dtype=sim.dtype)
weights[0, : ports[0].shape[1]] = 1.0
weights[1, ports[0].shape[1] :] = 1.0
objective = intensity(sim, FREQUENCIES, weights)

# --------------------------------------- source --------------------------------------
t = np.linspace(0, (N - 1) * dt, N)
line = np.arange(-0.5 * GUIDE1, 0.5 * GUIDE1, dx[1])
# broadband, so one run carries both design wavelengths; scaled by the along-line spacing
signal = ricker(t, AMPLITUDE * dx[1], 0.5 * sum(FREQUENCIES))
source = point_source(sim, [shift((SOURCE_X, 0.5 * HEIGHT + y)) for y in line], signal)

# ------------------------------------ optimization -----------------------------------
density_filter = DensityFilter(RMIN / min(dx), sim.Nx_padded, dtype=sim.dtype)
projection = Projection(BETA0, ETA)
beta_of = lambda i: min(BETA0 * BETA_GROWTH ** (i // BETA_STEP), BETA_MAX)
info = {}


def adjoint(*args):
    """`reconstruction_sensitivity`, keeping the drift its reverse march reports."""
    out = reconstruction_sensitivity(*args)
    info.update(out[3])
    return out


def physical(variables: cpt.NDArray) -> tuple[cpt.NDArray, cpt.NDArray]:
    """Filter then project the design variables: (permittivity field, filtered field)."""
    filtered = density_filter(variables)
    return cp.maximum(projection(filtered) * region, waveguides), filtered


def transmitted(indicator: cpt.NDArray) -> np.ndarray:
    """The (frequency, port) intensity table, which is what says the device separates."""
    _, um = simulate(sim, source, indicator, sensors=sensors)
    table = np.empty((len(FREQUENCIES), len(ports)))
    for i in range(len(FREQUENCIES)):
        for j in range(len(ports)):
            mask = cp.zeros_like(weights)
            mask[i] = weights[j]
            table[i, j] = intensity(sim, FREQUENCIES, mask)(um)[0]
    return table


variables = cp.where(region, DESIGN_START, 0.0).astype(sim.dtype)
optimizer = Adam(lr=LEARNING_RATE)
history = []

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERATIONS):
    projection.set(beta=beta_of(iteration))
    design, filtered = physical(variables)
    cost, gradient = response_gradient(
        sim, source, design, sensors, objective, adjoint=adjoint
    )

    # chain rule: permittivity -> projection -> filter -> variables
    gradient = density_filter.grad(
        variables, projection.grad(filtered, gradient * region)
    )
    variables = cp.clip(optimizer.step(variables, -gradient * region), 0.0, 1.0)

    history.append(cost)
    print(
        f"iteration {iteration}: routed {cost:.4e}  "
        f"{10 * math.log10(cost / history[0]):+.2f} dB  beta {projection.beta:.0f}  "
        f"drift {info['drift']:.1e}"
    )
cp.cuda.Stream.null.synchronize()
print(f"{ITERATIONS:d} iterations of {N:d} steps: {time.time() - tic:.1f}s")

# ------------------------------------- evaluation ------------------------------------
projection.set(beta=beta_of(ITERATIONS))
design, _ = physical(variables)
final = cp.maximum(threshold(design, THRESHOLD, dtype=sim.dtype), waveguides)

cp.cuda.Stream.null.synchronize()
table = transmitted(final)
wavefield, _ = simulate(sim, source, final, sensors=sensors)
cp.cuda.Stream.null.synchronize()

# the thresholded design is the one that can be built, so it is the one that is reported
print(f"\nrouted {history[0]:.4e} -> {table[0, 0] + table[1, 1]:.4e} thresholded")
print(f"{'':>10}{'port 1':>14}{'port 2':>14}{'isolation':>12}")
for i, f in enumerate(FREQUENCIES):
    other = table[i, 1 - i]
    print(
        f"{1.0 / f:9.3f} {table[i, 0]:>13.4e} {table[i, 1]:>13.4e} "
        f"{10 * math.log10(table[i, i] / other):>10.2f} dB"
    )
print(f"\tnon-discreteness {non_discreteness(design, region):.3f}")
print(f"\tsilicon fraction {float(design[region].sum()) / int(region.sum()):.3f}")

# ----------------------------------- postprocessing ----------------------------------
scale = 0.5 * float(cp.max(cp.abs(wavefield[domain]))) + 1e-30
low = (region.nonzero()[0].min().get(), region.nonzero()[1].min().get())
high = (region.nonzero()[0].max().get(), region.nonzero()[1].max().get())

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].semilogy(history, "k")
show(axes[1], field=wavefield[domain], indicator=final[domain], scale=scale)
outline(axes[1], low, high, origin=pad + 1, color="k")
axes[1].set_aspect("equal")
axes[1].axis("off")
fig.tight_layout()
plt.show()
