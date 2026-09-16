import math
import time
from dataclasses import replace

import cupy as cp
import cupy.typing as cpt
import matplotlib.pyplot as plt
import numpy as np

from cuwave.boundary import pad_for_sponge, sponge
from cuwave.evals import non_discreteness
from cuwave.geometry import box
from cuwave.maxwell import MagneticWave
from cuwave.optimization import Adam
from cuwave.postprocessing import markers, outline, show
from cuwave.regularization import DensityFilter, Projection
from cuwave.sensitivity import reconstruction_sensitivity
from cuwave.signals import ricker
from cuwave.utils import (
    Sensors,
    intensity,
    point_source,
    response,
    response_gradient,
    threshold,
)
from cuwave.wave import grid_coords, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 2  # the design enters the stiffness, whose transpose is exact only here
PRECISION = "float32"
POINTS_PER_WAVELENGTH = 20  # in silicon, the short wavelength of the two
SAFETY = 0.95

# physics, nondimensional in the design wavelength: lambda = c = eps_air = mu = 1
INDEX_SI = 3.48  # dispersion and loss are not modelled, so one index serves every band
PERMITTIVITY1, PERMITTIVITY2 = 1.0, INDEX_SI**2
PERMEABILITY = 1.0
FREQUENCIES = (1.0,)  # (0.917, 1.0, 1.1) is the broadband lens of the paper's case 4
AMPLITUDE = 1.0
T = 40.0

# geometry (design wavelengths), from table 1 of Christiansen & Sigmund 2021 over 550 nm
WIDTH, HEIGHT = 8.727, 3.302
SUBSTRATE = 0.227  # silicon the lens stands on, carried on through the lower layer
DESIGN_WIDTH, DESIGN_HEIGHT = 5.455, 0.455
FOCAL_DISTANCE = 1.321  # above the design, giving a numerical aperture of 0.9
ENVELOPE = 2.727  # gaussian width of the incident beam
SOURCE_DEPTH = 0.15  # below the substrate top, so the beam enters through the silicon

# boundary
THICKNESS = 2.0  # sponge thickness in design wavelengths
SPONGE_DECAY = 4.3  # decay per wavelength travelled, so the layer follows the timestep

# optimization
ITERATIONS = 60
LEARNING_RATE = 0.05
DESIGN_START = 0.5  # range [0, 1]

# regularization
RMIN = 0.05  # filter radius in design wavelengths
ETA = 0.5
BETA0, BETA_GROWTH, BETA_STEP, BETA_MAX = 1.0, 2.0, 10, 64.0

# evaluation
THRESHOLD = 0.5  # the projection maps onto [0, 1], so its midpoint is the cut

# --------------------------------------- setup ---------------------------------------
dx = (1.0 / (INDEX_SI * POINTS_PER_WAVELENGTH),) * 2
resolution = tuple(int(length / dx[d]) + 3 for d, length in enumerate((WIDTH, HEIGHT)))
Nx, pad, origin, domain = pad_for_sponge(resolution, dx, THICKNESS)

dt = SAFETY * stable_dt(dx, 1.0, SPACE_ORDER)  # vacuum is the fast phase, not silicon
N = math.ceil(T / dt)
shift = lambda point: tuple(o + c for o, c in zip(origin, point))

sim = MagneticWave(
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
    f"{1.0 / (INDEX_SI * dx[0]):.0f} points per silicon wavelength"
)

coords = grid_coords(Nx, dx, dtype=sim.dtype)
# the substrate runs on through the lower layer, so the beam meets no step entering it
substrate = coords[1] <= origin[1] + SUBSTRATE
design_center = (0.5 * WIDTH, SUBSTRATE + 0.5 * DESIGN_HEIGHT)
region = box(coords, shift(design_center), (DESIGN_WIDTH, DESIGN_HEIGHT))
focus = shift((0.5 * WIDTH, SUBSTRATE + DESIGN_HEIGHT + FOCAL_DISTANCE))
print(f"{int(region.sum())} design nodes, focus {FOCAL_DISTANCE:.2f} wavelengths up")

# --------------------------------------- source --------------------------------------
t = np.linspace(0, (N - 1) * dt, N)
line = np.linspace(0.0, WIDTH, resolution[0] - 2)
envelope = np.exp(-(((line - 0.5 * WIDTH) / ENVELOPE) ** 2))
# scaled by the spacing along the line, so the sheet current does not move with the grid
signal = np.outer(ricker(t, AMPLITUDE * dx[0], min(FREQUENCIES)), envelope)
source = point_source(sim, [shift((x, SUBSTRATE - SOURCE_DEPTH)) for x in line], signal)

receivers = Sensors(sim, [focus])
objective = receivers.objective(intensity(sim, FREQUENCIES))

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
    return cp.maximum(projection(filtered) * region, substrate), filtered


variables = cp.where(region, DESIGN_START, 0.0).astype(sim.dtype)
optimizer = Adam(lr=LEARNING_RATE)
history = []

# the empty lens is what the focal gain is quoted against, so the units drop out
bare, _ = response(sim, source, substrate.astype(sim.dtype), receivers.nodes, objective)
print(f"bare substrate focuses {bare:.4e}")

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERATIONS):
    projection.set(beta=beta_of(iteration))
    design, filtered = physical(variables)
    cost, gradient = response_gradient(
        sim, source, design, receivers.nodes, objective, adjoint=adjoint
    )

    # chain rule: permittivity -> projection -> filter -> variables
    gradient = density_filter.grad(
        variables, projection.grad(filtered, gradient * region)
    )
    variables = cp.clip(optimizer.step(variables, -gradient * region), 0.0, 1.0)

    history.append(cost)
    print(
        f"iteration {iteration}: focus {cost:.4e}  "
        f"{10 * math.log10(cost / bare):+.2f} dB  beta {projection.beta:.0f}  "
        f"drift {info['drift']:.1e}"
    )
cp.cuda.Stream.null.synchronize()
print(f"{ITERATIONS:d} iterations of {N:d} steps: {time.time() - tic:.1f}s")

# ------------------------------------- evaluation ------------------------------------
projection.set(beta=beta_of(ITERATIONS))
design, _ = physical(variables)
final = cp.maximum(threshold(design, THRESHOLD, dtype=sim.dtype), substrate)

cp.cuda.Stream.null.synchronize()
grey_cost, _ = response(sim, source, design, receivers.nodes, objective)
final_cost, wavefield = response(sim, source, final, receivers.nodes, objective)
cp.cuda.Stream.null.synchronize()

# the thresholded design is the one that can be built, so it is the one that is reported
print(
    f"\nfocus {bare:.4e} bare -> {grey_cost:.4e} grey -> {final_cost:.4e} thresholded  "
    f"({10 * math.log10(final_cost / bare):+.2f} dB)"
)
print(
    f"\tdiscretization price {10 * math.log10(final_cost / grey_cost):+.2f} dB  "
    f"at non-discreteness {non_discreteness(design, region):.3f}"
)
print(f"\tsilicon fraction {float(design[region].sum()) / int(region.sum()):.3f}")

# ----------------------------------- postprocessing ----------------------------------
scale = 0.5 * float(cp.max(cp.abs(wavefield[domain]))) + 1e-30
low = (region.nonzero()[0].min().get(), region.nonzero()[1].min().get())
high = (region.nonzero()[0].max().get(), region.nonzero()[1].max().get())

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].semilogy(history, "k")
show(axes[1], field=wavefield[domain], indicator=final[domain], scale=scale)
outline(axes[1], low, high, origin=pad + 1, color="k")
markers(axes[1], focus, dx=dx, origin=pad + 1, nodes=6, color="k")
axes[1].set_aspect("equal")
axes[1].axis("off")
fig.tight_layout()
plt.show()
