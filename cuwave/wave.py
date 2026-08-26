from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
import numpy as np

from .boundary import canonical_boundary, define_boundary
from .stencils import preamble, weights

KERNEL_PATH = Path(__file__).parent / "kernels" / "wave.cu"


# ------------------------------------- utilities -------------------------------------
def stable_dt(dx, wavespeed, space_order=2):
    """CFL-stable timestep for an explicit scheme with grid spacing `dx`, given `wavespeed` and finite difference `space_order`."""
    lam = float(np.abs(weights(space_order // 2)).sum())
    return 2.0 / (wavespeed * float(np.sqrt(lam * sum(1.0 / d**2 for d in dx))))


# -------------------------------------- helpers --------------------------------------
def padded_shape(Nx):
    """Pad the fastest (last) axis of `Nx` up to a multiple of 32, for coalesced access."""
    return (*Nx[:-1], ((Nx[-1] + 31) // 32) * 32)


def mirror_ghosts(sim, field):
    """Mirror `field`'s ghost layer onto its second-interior node on every axis, for a reflecting wall."""
    for d in range(sim.ndim):
        for ghost, mirror in ((0, 2), (sim.Nx[d] - 1, sim.Nx[d] - 3)):
            dst = [slice(None)] * sim.ndim
            dst[d] = ghost
            src = [slice(None)] * sim.ndim
            src[d] = mirror
            field[tuple(dst)] = field[tuple(src)]
    return field


def grid_coords(Nx, dx, dtype=cp.float64):
    """Padded grid coordinates for shape `Nx` at spacing `dx`, with node 1 at the origin."""
    # indices 0 and -1 are ghost nodes outside the domain
    axes = [(cp.arange(n, dtype=dtype) - 1) * d for n, d in zip(padded_shape(Nx), dx)]
    return cp.meshgrid(*axes, indexing="ij")


@dataclass
class Source:
    position: cpt.NDArray[cp.int32]  # (ndim, num_sources) grid indices
    signal: cpt.NDArray  # (N, num_sources) time series


def flatten_indices(sim, position):
    """Collapse (ndim, num) grid indices `position` into flat indices of the padded array."""
    lin = cp.zeros(position.shape[1], dtype=cp.int32)
    for d in range(sim.ndim):
        lin += position[d] * cp.int32(sim.strides[d])
    return lin


# -------------------------------- discretization setup -------------------------------
@dataclass
class Simulation:
    """Grid, timestepping, and compile-time configuration shared by all wave equations."""

    Nx: tuple[int, ...]  # logical grid points per axis (incl. ghost cells)
    dx: tuple[float, ...]
    N: int  # number of time steps
    dt: float
    threads: tuple[int, ...]  # threads per block, per axis
    precision: str = "float32"  # "float32" or "float64"
    space_order: int = 2  # finite difference order: any even number
    boundary: tuple = None  # ((low, high),) per axis; None is Neumann everywhere

    compile_flags = ()  # extra nvcc -D flags

    def __post_init__(self):
        """Derive `ndim`, padded shape, strides, dtype, and canonical `boundary` from the raw fields."""
        self.ndim = len(self.Nx)
        self.Nx_padded = padded_shape(self.Nx)
        # C-contiguous strides over the padded shape (last axis has unit stride)
        strides = [1] * self.ndim
        for d in range(self.ndim - 2, -1, -1):
            strides[d] = strides[d + 1] * self.Nx_padded[d + 1]
        self.strides = tuple(strides)
        self.dtype = cp.float32 if self.precision == "float32" else cp.float64
        self.boundary = canonical_boundary(self.boundary, self.ndim)
        if self.space_order % 2 != 0 or self.space_order < 2:
            raise ValueError("space_order must be an even integer >= 2")


@dataclass
class PressureWave(Simulation):
    """Scalar wave equation base: nodal `stiff`/`minv` material fields, optional damping."""

    damped = False  # or damping field
    derive_inertia = False  # set where m == k (rho scaling): minv derived from stiff

    @property
    def compile_flags(self):
        """`-DUSE_DAMPING` when a damping field is set, else no extra flags."""
        return ("-DUSE_DAMPING",) if self.damped else ()

    def build_materials(self, indicator, damping=None):
        """Turn `indicator` (and optional `damping`) into the kernel's ghost-mirrored material dict."""
        self.damped = damping is not None
        stiff, minv = self.parametrization(indicator)
        # mirrored in place, so the ghost ring of a caller-supplied indicator is
        # normalised to the value the reflecting wall implies (see mirror_ghosts)
        mat = {"stiff": mirror_ghosts(self, stiff)}
        mat["minv"] = None if minv is None else mirror_ghosts(self, minv)
        if self.damped:
            mat["damping"] = damping
        return mat

    def step_kernel_args(self, mat):
        """Material and damping arguments for the finite-difference step kernel."""
        minv = mat["stiff"] if self.derive_inertia else mat["minv"]
        args = (mat["stiff"], minv, np.int32(self.derive_inertia))
        if self.damped:
            args += (mat["damping"], self.dtype(self.dt))
        return args

    def excitation_weights(self, mat, lin_index):
        """Source weights ``dt**2 / inertia`` at `lin_index`, without materializing inertia over the grid."""
        field = mat["stiff"] if self.derive_inertia else mat["minv"]
        weight = field.ravel()[lin_index]
        if self.derive_inertia:
            weight = 1.0 / weight
        return (self.dtype(self.dt**2 * self.source_factor()) * weight).astype(
            self.dtype
        )


@dataclass
class ScalarWave(PressureWave):
    """Constant-speed scalar wave equation, parametrized by a density-scaling indicator gamma."""

    wavespeed: float = None  # background wave speed c0
    density: float = None  # background density rho0

    derive_inertia = True  # gamma scales inertia and stiffness alike

    def __post_init__(self):
        """Validate `wavespeed` and `density` are set, on top of `Simulation.__post_init__`."""
        super().__post_init__()
        if self.wavespeed is None or self.density is None:
            raise ValueError("ScalarWave requires wavespeed and density")

    def parametrization(self, indicator):
        """`indicator` is the density-scaling field gamma; `minv` is left to be derived from it."""
        return indicator, None

    def parametrization_jacobian(self):
        """Both mass and stiffness coefficients are gamma itself, so both derivatives are 1."""
        return 1.0, 1.0

    def step_factors(self):
        """Per-axis finite-difference step factors ``2 * c0**2 * dt**2 / dx**2``."""
        return [
            self.dtype(2.0 * self.wavespeed**2 * self.dt**2 / dxk**2) for dxk in self.dx
        ]

    def source_factor(self):
        """Source scaling ``1 / rho0``."""
        return 1.0 / self.density


@dataclass
class AcousticWave(PressureWave):
    """Two-phase acoustic wave equation (TATO), gamma interpolating between air and solid."""

    # TATO material constants (gamma = 0 -> air, gamma = 1 -> solid)
    rho1: float = None
    rho2: float = None
    kappa1: float = None
    kappa2: float = None

    def __post_init__(self):
        """Validate the four material constants are set, on top of `Simulation.__post_init__`."""
        super().__post_init__()
        if None in (self.rho1, self.rho2, self.kappa1, self.kappa2):
            raise ValueError("AcousticWave requires rho1, rho2, kappa1, kappa2")

    def parametrization(self, indicator):
        """`indicator` interpolates inverse density and inverse bulk modulus between phase 1 and 2."""
        # gamma interpolates the inverse density and the inverse bulk modulus;
        # the kernel wants 1 / rho, so rho is never formed
        rho_inv = 1 / self.rho1 + indicator * (1 / self.rho2 - 1 / self.rho1)
        kappa_inv = 1 / self.kappa1 + indicator * (1 / self.kappa2 - 1 / self.kappa1)
        return rho_inv, 1 / kappa_inv

    def parametrization_jacobian(self):
        """Derivatives of (mass, stiff) with respect to gamma; both coefficients are affine in it."""
        # both coefficients are affine in gamma once written as (mass, stiff):
        # mass = 1 / kappa and stiff = 1 / rho are the two the interpolation is
        # linear in, which is why kappa itself is inverted above
        return (
            1 / self.kappa2 - 1 / self.kappa1,
            1 / self.rho2 - 1 / self.rho1,
        )

    def step_factors(self):
        """Per-axis finite-difference step factors ``2 * dt**2 / dx**2``."""
        return [self.dtype(2.0 * self.dt**2 / dxk**2) for dxk in self.dx]

    def source_factor(self):
        """Source scaling, unscaled since rho is already folded into `parametrization`."""
        return 1.0


# ----------------------------------- kernel helpers ----------------------------------


def compile_kernels(sim, path=KERNEL_PATH):
    """Compile the kernel source at `path` for `sim`, with the stencil table injected as source."""
    options = ["--use_fast_math", f"-DNDIM={sim.ndim}", *sim.compile_flags]
    if sim.precision == "float32":
        options.append("-DUSE_FLOAT")
    # the stencil table is injected as source, so the CuPy module cache keys on
    # the order without it having to appear in the compile options
    code = preamble(sim.space_order) + Path(path).read_text()
    return cp.RawModule(code=code, options=tuple(options))


def grid_block(sim):
    """CUDA (grid, block) dimensions for `sim`, fastest axis mapped to x."""
    # map the fastest axis to grid/block x, the next to y, the next to z
    extent = sim.Nx_padded
    block = tuple(sim.threads[::-1])
    grid = tuple(
        (extent[d] + sim.threads[d] - 1) // sim.threads[d] for d in range(sim.ndim)
    )[::-1]
    return grid, block


def axis_geometry(sim, factors):
    """Interleave `factors` with axis extents and strides, in the layout the step kernel expects."""
    # kernel args after the material arrays: f0, N0, [f1, N1, s0], [f2, N2, s1]
    geom = [factors[0], sim.Nx[0]]
    for d in range(1, sim.ndim):
        geom += [factors[d], sim.Nx[d], sim.strides[d - 1]]
    return geom


# -------------------------------- simulation functions -------------------------------


def define_step_method(sim, kernels, mat):
    """Closure launching the finite-difference step kernel over (u0, u1, u2)."""
    fd_kernel = kernels.get_function("fd_kernel")
    grid, block = grid_block(sim)
    args = [
        None,
        None,
        None,
        *sim.step_kernel_args(mat),
        *axis_geometry(sim, sim.step_factors()),
    ]

    def fd_step(u0, u1, u2):
        args[0], args[1], args[2] = u0, u1, u2
        fd_kernel(grid, block, args)
        return u2

    return fd_step


def define_excitation(sim, position, kernels, mat):
    """Closure injecting `signal` at `position` into `u` at timestep `t_index`."""
    excitation_kernel = kernels.get_function("excitation_kernel")
    threads = 256
    num_sources = position.shape[1]
    blocks = (num_sources + threads - 1) // threads
    lin_index = flatten_indices(sim, position)
    weight = sim.excitation_weights(mat, lin_index)
    # the whole (N, num_sources) record plus a row offset, not a row view
    args = [None, None, np.int32(0), lin_index, np.int32(num_sources), weight]

    def excitation_step(u, signal, t_index):
        args[0], args[1] = u, signal
        args[2] = np.int32(t_index * num_sources)
        excitation_kernel((blocks,), (threads,), args)
        return u

    return excitation_step


def define_get_signal(sim, sensors, kernels):
    """Closure writing row `t_index` of the `(N, num_sensors)` record `um` from `u`."""
    get_signal_kernel = kernels.get_function("get_signal_kernel")
    threads = 256
    num_sensors = sensors.shape[1]
    blocks = (num_sensors + threads - 1) // threads
    lin_index = flatten_indices(sim, sensors)
    args = [None, None, np.int32(0), lin_index, np.int32(num_sensors)]

    # writes row t of the whole (N, num_sensors) record, so the caller never slices
    def get_signal_step(u, um, t_index):
        args[0], args[1] = u, um
        args[2] = np.int32(t_index * num_sensors)
        get_signal_kernel((blocks,), (threads,), args)
        return um

    return get_signal_step


def simulate(
    sim,
    source,
    indicator,
    damping=None,
    sensors=None,
    record_every=None,
):
    """Run `sim` forward under `source` and material `indicator`, optionally recording sensor traces and field snapshots."""
    U = cp.zeros((2, *sim.Nx_padded), dtype=sim.dtype)
    u0, u1 = U[0], U[1]

    mat = sim.build_materials(indicator, damping)
    kernels = compile_kernels(sim)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    if sensors is not None:
        get_signal = define_get_signal(sim, sensors, kernels)
        um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)
    interior = tuple(slice(0, n) for n in sim.Nx)
    field = lambda u: u[interior]
    snapshots = []

    for t in range(sim.N):
        u0 = fd_step(u0, u1, u0)
        u0 = excitation_step(u0, source.signal, t)
        u0 = bc_step(u0)
        u1, u0 = u0, u1
        if sensors is not None:
            get_signal(u1, um, t)
        if record_every is not None and t % record_every == 0:
            snapshots.append(field(u1).get())

    if sensors is not None and record_every is not None:
        return field(u1), um, np.stack(snapshots)
    if sensors is not None:
        return field(u1), um
    if record_every is not None:
        return field(u1), np.stack(snapshots)
    return field(u1)
