"""Scalar and acoustic wave equations on a padded finite-difference grid.

`Simulation` holds the grid and the compile-time configuration, `PressureWave` adds the
nodal material fields, and `ScalarWave` / `AcousticWave` supply the parametrization that
turns an indicator into those fields. The `define_*` factories bind compiled kernels to
one such configuration, and `simulate` loops over the closures they return.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
import numpy as np
import numpy.typing as npt

from .boundary import Neumann, canonical_boundary, define_boundary
from .stencils import preamble, weights

KERNEL_PATH = Path(__file__).parent / "kernels" / "wave.cu"
SENSITIVITY_PATH = Path(__file__).parent / "kernels" / "wave_sensitivity.cu"


# ------------------------------------- utilities -------------------------------------
def stable_dt(dx: tuple[float, ...], wavespeed: float, space_order: int = 2) -> float:
    """CFL-stable timestep for an explicit scheme with grid spacing `dx`."""
    lam = float(np.abs(weights(space_order // 2)).sum())
    return 2.0 / (wavespeed * float(np.sqrt(lam * sum(1.0 / d**2 for d in dx))))


# -------------------------------------- helpers --------------------------------------
def padded_shape(Nx: tuple[int, ...]) -> tuple[int, ...]:
    """Pad the fastest (last) axis of `Nx` up to a multiple of 32, for coalesced access."""
    return (*Nx[:-1], ((Nx[-1] + 31) // 32) * 32)


def mirror_ghosts(sim: Simulation, field: cpt.NDArray) -> cpt.NDArray:
    """Mirror `field`'s ghost layer onto its second-interior node, for homogeneous Neumann."""
    for d in range(sim.ndim):
        for ghost, mirror in ((0, 2), (sim.Nx[d] - 1, sim.Nx[d] - 3)):
            dst = [slice(None)] * sim.ndim
            dst[d] = ghost
            src = [slice(None)] * sim.ndim
            src[d] = mirror
            field[tuple(dst)] = field[tuple(src)]
    return field


def grid_coords(
    Nx: tuple[int, ...], dx: tuple[float, ...], dtype: npt.DTypeLike = cp.float64
) -> list[cpt.NDArray]:
    """Padded grid coordinates for shape `Nx` at spacing `dx`, with node 1 at the origin."""
    # indices 0 and -1 are ghost nodes outside the domain
    axes = [(cp.arange(n, dtype=dtype) - 1) * d for n, d in zip(padded_shape(Nx), dx)]
    return cp.meshgrid(*axes, indexing="ij")


@dataclass
class Source:
    position: cpt.NDArray[cp.int32]  # (ndim, num_sources) grid indices
    signal: cpt.NDArray  # (N, num_sources) time series


# W is the cell volume a node owns; docs/sensitivity.md derives it
def apply_cell_weights(sim: Simulation, field: cpt.NDArray) -> cpt.NDArray:
    """In-place multiply of `field` by the cell weights W."""
    # axes in sequence, so a corner compounds to 1/4 (1/8 in 3D)
    for d in range(sim.ndim):
        for index in (1, sim.Nx[d] - 2):
            face = [slice(None)] * sim.ndim
            face[d] = index
            field[tuple(face)] *= 0.5
    return field


def sensor_cell_weights(sim: Simulation, sensors: cpt.NDArray[cp.int32]) -> cpt.NDArray:
    """Cell weights W at the sensor nodes only, as a (num_sensors,) vector."""
    # the nested loop of apply_cell_weights, so the two agree on a degenerate axis too
    w = cp.ones(sensors.shape[1], dtype=sim.dtype)
    for d in range(sim.ndim):
        for index in (1, sim.Nx[d] - 2):
            w = cp.where(grid_rows(sim, sensors)[d] == index, w * 0.5, w)
    return w


def grid_rows(
    sim: Simulation, position: cpt.NDArray[cp.int32]
) -> cpt.NDArray[cp.int32]:
    """The `ndim` spatial rows of `position`, dropping a leading component row."""
    return position[-sim.ndim :]


def flatten_indices(
    sim: Simulation, position: cpt.NDArray[cp.int32]
) -> cpt.NDArray[cp.int32]:
    """Collapse (node_rows, num) indices `position` into flat indices of the field.

    A vector unknown takes a leading component row, folded in as
    `component * comp_stride`, so the gather and scatter kernels stay scalar.
    """
    if position.shape[0] != sim.node_rows:
        raise ValueError(
            f"position needs {sim.node_rows} rows for ncomp={sim.ncomp} in "
            f"{sim.ndim}D, not {position.shape[0]}"
        )
    lin = cp.zeros(position.shape[1], dtype=cp.int32)
    for d in range(sim.ndim):
        lin += grid_rows(sim, position)[d] * cp.int32(sim.strides[d])
    if sim.ncomp > 1:
        lin += position[0] * cp.int32(sim.comp_stride)
    return lin


# -------------------------------- discretization setup -------------------------------
@dataclass
class Simulation:
    """Grid, timestepping, and compile-time configuration shared by all wave equations."""

    Nx: tuple[int, ...]  # logical grid points per axis (incl. ghost nodes)
    dx: tuple[float, ...]
    N: int  # number of time steps
    dt: float
    threads: tuple[int, ...]  # threads per block, per axis
    precision: str = "float32"  # "float32" or "float64"
    space_order: int = 2  # finite difference order: any even number
    boundary: tuple = None  # ((low, high),) per axis; None is the equation's default
    damping: cpt.NDArray | None = None  # nodal field d, or None for a lossless operator

    @property
    def compile_flags(self) -> tuple[str, ...]:
        """`-DUSE_DAMPING` when a damping field is set, else no extra flags."""
        return ("-DUSE_DAMPING",) if self.damping is not None else ()

    kernel_path = KERNEL_PATH  # the forward source this equation compiles
    sensitivity_path = SENSITIVITY_PATH  # and the adjoint one
    default_boundary = Neumann  # what `boundary=None` means for this equation

    @property
    def ncomp(self) -> int:
        """Field components per node: 1 for a scalar unknown, `ndim` for a vector one."""
        return 1

    def __post_init__(self) -> None:
        """Derive `ndim`, padded shape, strides, dtype, and canonical `boundary`."""
        self.ndim = len(self.Nx)
        self.Nx_padded = padded_shape(self.Nx)
        # C-contiguous strides over the padded shape (last axis has unit stride)
        strides = [1] * self.ndim
        for d in range(self.ndim - 2, -1, -1):
            strides[d] = strides[d + 1] * self.Nx_padded[d + 1]
        self.strides = tuple(strides)
        self.dtype = cp.float32 if self.precision == "float32" else cp.float64
        self.boundary = canonical_boundary(
            self.boundary, self.ndim, self.default_boundary
        )
        if self.space_order % 2 != 0 or self.space_order < 2:
            raise ValueError("space_order must be an even integer >= 2")
        self.comp_stride = int(np.prod(self.Nx_padded))
        self.node_rows = self.ndim + (self.ncomp > 1)
        self.field_shape = (
            self.Nx_padded if self.ncomp == 1 else (self.ncomp, *self.Nx_padded)
        )
        # flatten_indices accumulates in int32, so the whole field has to address in it
        if self.ncomp * self.comp_stride >= 2**31:
            raise ValueError(
                f"{self.ncomp} x {self.comp_stride} nodes overflow the int32 flat "
                f"index; coarsen the grid"
            )


@dataclass
class PressureWave(Simulation):
    """Scalar wave equation base: nodal `stiff`/`minv` material fields, optional damping."""

    derive_inertia = False  # set where m == k (rho scaling): minv derived from stiff

    def build_materials(self, indicator: cpt.NDArray) -> dict:
        """Turn `indicator` into the kernel's material dict, `damping` included."""
        stiff, minv = self.parametrization(indicator)
        # mirrored in place, so a caller's ghost ring is normalised to Neumann
        mat = {"stiff": mirror_ghosts(self, stiff)}
        mat["minv"] = None if minv is None else mirror_ghosts(self, minv)
        if self.damping is not None:
            mat["damping"] = self.damping
        return mat

    def inverse_inertia(self, indicator: cpt.NDArray) -> cpt.NDArray:
        """Nodal `1 / m` for `indicator`, derived from the stiffness where `m == k`."""
        stiff, minv = self.parametrization(indicator)
        return 1.0 / stiff if minv is None else minv

    def step_kernel_args(self, mat: dict) -> tuple:
        """Material and damping arguments for the finite-difference step kernel."""
        minv = mat["stiff"] if self.derive_inertia else mat["minv"]
        args = (mat["stiff"], minv, np.int32(self.derive_inertia))
        if self.damping is not None:
            args += (mat["damping"], self.dtype(self.dt))
        return args

    gradient_names = ("mass", "stiff")  # the material fields the adjoint differentiates

    def gradient_fields(self, mat: dict) -> dict[str, cpt.NDArray]:
        """Zeroed accumulators the adjoint kernels add into, one per material field."""
        return {
            name: cp.zeros(self.Nx_padded, dtype=self.dtype)
            for name in self.gradient_names
        }

    def adjoint_weights(self, sensors: cpt.NDArray[cp.int32]) -> cpt.NDArray:
        """Divisor the adjoint source carries: the cell weights W over the source factor."""
        return sensor_cell_weights(self, sensors) * self.source_factor()

    def finalize_gradients(self, grads: dict, kernels: cp.RawModule) -> dict:
        """Weight the accumulators by W once the time loop is done."""
        for field in grads.values():
            apply_cell_weights(self, field)
        return grads

    def define_gradient(
        self, kernels: cp.RawModule, mat: dict, grads: dict
    ) -> Callable:
        """Closure accumulating both gradient densities from a forward triplet and `l1`."""
        # one kernel for both gradients: a launch costs more host time than either body
        gradient_kernel = kernels.get_function("gradient_kernel")
        grid, block = grid_block(self)
        # the operator without the dt^2 the step folds into it: L, not dt^2 L
        factors = [self.dtype(float(f) / self.dt**2) for f in self.step_factors()]
        geom = [factors[0], self.Nx[0]]
        for d in range(1, self.ndim):
            geom += [factors[d], self.Nx[d], self.strides[d - 1]]
        args = [grads["mass"], grads["stiff"], None, None, None, None] + [
            mat["stiff"],
            self.dtype(1.0 / self.dt**2),
            *geom,
        ]

        def gradient_step(u0, u1, u2, l1):
            args[2], args[3], args[4], args[5] = u0, u1, u2, l1
            gradient_kernel(grid, block, args)

        return gradient_step

    def define_frechet(
        self, kernels: cp.RawModule, accs: dict, sign: float
    ) -> Callable:
        """Closure accumulating both Frechet densities of one field triplet, times `sign`."""
        # sign is fixed per pass, so it is folded into the factors rather than recomputed
        frechet_kernel = kernels.get_function("frechet_kernel")
        grid, block = grid_block(self)
        # the stiffness density enters negated, so the epilogue scales both alike
        geom = [self.dtype(sign / (2.0 * self.dt) ** 2)]
        for d in range(self.ndim):
            geom.append(self.dtype(-sign / (2.0 * self.dx[d]) ** 2))
            geom.append(self.Nx[d])
            if d:
                geom.append(self.strides[d - 1])
        args = [accs["mass"], accs["stiff"], None, None, None] + geom

        def frechet_step(u0, u1, u2):
            args[2], args[3], args[4] = u0, u1, u2
            frechet_kernel(grid, block, args)

        return frechet_step

    def excitation_weights(
        self, mat: dict, lin_index: cpt.NDArray[cp.int32]
    ) -> cpt.NDArray:
        """Source weights `dt**2 / inertia` at `lin_index`, never forming inertia on the grid."""
        field = mat["stiff"] if self.derive_inertia else mat["minv"]
        weight = field.ravel()[lin_index]
        if self.derive_inertia:
            weight = 1.0 / weight
        if self.damping is not None:
            # the damped update divides by 1 + beta, so the added source has to share it
            beta = 0.5 * weight * mat["damping"].ravel()[lin_index] * self.dt
            weight = weight / (1.0 + beta)
        return (self.dtype(self.dt**2 * self.source_factor()) * weight).astype(
            self.dtype
        )


@dataclass
class ScalarWave(PressureWave):
    """Constant-speed scalar wave equation, parametrized by a density-scaling indicator gamma."""

    wavespeed: float = None  # background wave speed c0
    density: float = None  # background density rho0

    derive_inertia = True  # gamma scales inertia and stiffness alike

    def __post_init__(self) -> None:
        """Validate `wavespeed` and `density` are set, on top of `Simulation.__post_init__`."""
        super().__post_init__()
        if self.wavespeed is None or self.density is None:
            raise ValueError("ScalarWave requires wavespeed and density")

    def parametrization(
        self, indicator: cpt.NDArray
    ) -> tuple[cpt.NDArray, cpt.NDArray | None]:
        """`indicator` is the density-scaling field gamma; `minv` is left to be derived from it."""
        return indicator, None

    def parametrization_jacobian(self, indicator: cpt.NDArray) -> tuple:
        """Both mass and stiffness coefficients are gamma itself, so both derivatives are 1."""
        return 1.0, 1.0

    def step_factors(self) -> list:
        """Per-axis finite-difference step factors `2 * c0**2 * dt**2 / dx**2`."""
        return [
            self.dtype(2.0 * self.wavespeed**2 * self.dt**2 / dxk**2) for dxk in self.dx
        ]

    def source_factor(self) -> float:
        """Source scaling `1 / rho0`."""
        return 1.0 / self.density


@dataclass
class AcousticWave(PressureWave):
    """Two-phase acoustic wave equation (TATO), gamma interpolating between air and solid."""

    # TATO material constants (gamma = 0 -> air, gamma = 1 -> solid)
    rho1: float = None
    rho2: float = None
    kappa1: float = None
    kappa2: float = None

    def __post_init__(self) -> None:
        """Validate the four material constants are set, on top of `Simulation.__post_init__`."""
        super().__post_init__()
        if None in (self.rho1, self.rho2, self.kappa1, self.kappa2):
            raise ValueError("AcousticWave requires rho1, rho2, kappa1, kappa2")

    def parametrization(
        self, indicator: cpt.NDArray
    ) -> tuple[cpt.NDArray, cpt.NDArray]:
        """`indicator` interpolates inverse density and inverse bulk modulus between the phases."""
        # the kernel wants 1 / rho, so rho is never formed
        rho_inv = 1 / self.rho1 + indicator * (1 / self.rho2 - 1 / self.rho1)
        kappa_inv = 1 / self.kappa1 + indicator * (1 / self.kappa2 - 1 / self.kappa1)
        return rho_inv, 1 / kappa_inv

    def parametrization_jacobian(self, indicator: cpt.NDArray) -> tuple:
        """Derivatives of (mass, stiff) with respect to gamma; both coefficients are affine in it."""
        # affine in gamma as (mass, stiff) = (1 / kappa, 1 / rho), hence constant
        return (
            1 / self.kappa2 - 1 / self.kappa1,
            1 / self.rho2 - 1 / self.rho1,
        )

    def step_factors(self) -> list:
        """Per-axis finite-difference step factors `2 * dt**2 / dx**2`."""
        return [self.dtype(2.0 * self.dt**2 / dxk**2) for dxk in self.dx]

    def source_factor(self) -> float:
        """Source scaling, unscaled since rho is already folded into `parametrization`."""
        return 1.0


# ----------------------------------- kernel helpers ----------------------------------
def compile_kernels(sim: Simulation, path: Path | None = None) -> cp.RawModule:
    """Compile `path` for `sim`, defaulting to its own source, stencil table injected."""
    path = sim.kernel_path if path is None else path
    options = ["--use_fast_math", f"-DNDIM={sim.ndim}", *sim.compile_flags]
    if sim.precision == "float32":
        options.append("-DUSE_FLOAT")
    # injected as source, so the module cache keys on the order without a -D flag
    code = preamble(sim.space_order) + Path(path).read_text()
    return cp.RawModule(code=code, options=tuple(options))


def grid_block(sim: Simulation) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """CUDA (grid, block) dimensions for `sim`, fastest axis mapped to x."""
    # map the fastest axis to grid/block x, the next to y, the next to z
    extent = sim.Nx_padded
    block = tuple(sim.threads[::-1])
    grid = tuple(
        (extent[d] + sim.threads[d] - 1) // sim.threads[d] for d in range(sim.ndim)
    )[::-1]
    return grid, block


def axis_geometry(sim: Simulation, factors: list) -> list:
    """Interleave `factors` with axis extents and strides, in the layout the step kernel expects."""
    # kernel args after the material arrays: f0, N0, [f1, N1, s0], [f2, N2, s1]
    geom = [factors[0], sim.Nx[0]]
    for d in range(1, sim.ndim):
        geom += [factors[d], sim.Nx[d], sim.strides[d - 1]]
    return geom


# -------------------------------- simulation functions -------------------------------
def define_step_method(sim: Simulation, kernels: cp.RawModule, mat: dict) -> Callable:
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


def define_excitation(
    sim: Simulation,
    position: cpt.NDArray[cp.int32],
    kernels: cp.RawModule,
    mat: dict,
) -> Callable:
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


def define_get_signal(
    sim: Simulation, sensors: cpt.NDArray[cp.int32], kernels: cp.RawModule
) -> Callable:
    """Closure writing row `t_index` of the (N, num_sensors) record `um` from `u`."""
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


def define_set_signal(
    sim: Simulation, sensors: cpt.NDArray[cp.int32], kernels: cp.RawModule
) -> Callable:
    """Closure writing `u` at `sensors` back from row `t_index` of the record `um`."""
    set_signal_kernel = kernels.get_function("set_signal_kernel")
    threads = 256
    num_sensors = sensors.shape[1]
    blocks = (num_sensors + threads - 1) // threads
    lin_index = flatten_indices(sim, sensors)
    args = [None, None, np.int32(0), lin_index, np.int32(num_sensors)]

    # assignment rather than the atomicAdd of define_excitation, so it restores a state
    def set_signal_step(u, um, t_index):
        args[0], args[1] = u, um
        args[2] = np.int32(t_index * num_sensors)
        set_signal_kernel((blocks,), (threads,), args)
        return u

    return set_signal_step


def simulate(
    sim: Simulation,
    source: Source,
    indicator: cpt.NDArray,
    sensors: cpt.NDArray[cp.int32] | None = None,
    record_every: int | None = None,
) -> cpt.NDArray | tuple:
    """Run `sim` forward under `source` and material `indicator`.

    Args:
        sim: the simulation to step, which fixes the grid and the kernels compiled.
        source: the shot to inject, its signal an (N, num_sources) record.
        indicator: the design field the materials are built from.
        sensors: (ndim, num_sensors) grid indices to record at, or None for no record.
        record_every: snapshot the interior field every this many steps, or None.

    Returns:
        the final interior field, followed by the (N, num_sensors) record when
        `sensors` is given and the stacked host snapshots when `record_every` is.
    """
    U = cp.zeros((2, *sim.field_shape), dtype=sim.dtype)
    u0, u1 = U[0], U[1]

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    if sensors is not None:
        get_signal = define_get_signal(sim, sensors, kernels)
        um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)
    interior = (Ellipsis, *(slice(0, n) for n in sim.Nx))
    snapshots = []

    def field(u):
        return u[interior]

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
