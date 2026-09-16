"""The grid, the compile-time configuration, and the time loop every equation shares.

`Simulation` holds both and leaves the physics to a subclass in its own module
(`scalar.py`, `elastic.py`, `anisotropic.py`), which names its kernel sources and supplies
the material and factor hooks. The `define_*` factories bind compiled kernels to one such
configuration, and `simulate` loops over the closures they return.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
import numpy as np
import numpy.typing as npt

from .boundary import canonical_boundary, define_boundary
from .stencils import preamble, weights

# Voigt row order per dimension, as (k, l) strain pairs; the shear rows are PAIRS[d][d:]
PAIRS = {
    1: ((0, 0),),
    2: ((0, 0), (1, 1), (0, 1)),
    3: ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)),
}


# ------------------------------------- utilities -------------------------------------
def stable_dt(dx: tuple[float, ...], wavespeed: float, space_order: int = 2) -> float:
    """CFL-stable timestep for an explicit scheme with grid spacing `dx`."""
    lam = float(np.abs(weights(space_order // 2)).sum())
    return 2.0 / (wavespeed * float(np.sqrt(lam * sum(1.0 / d**2 for d in dx))))


def stable_timestep(
    sim: Simulation,
    indicator: cpt.NDArray,
    iterations: int = 60,
    safety: float = 0.95,
) -> float:
    """Largest stable timestep for `sim` under `indicator`, measured not estimated.

    The leapfrog is stable while the spectral radius of `dt**2 minv L` stays under 4, and
    `L` is symmetric with `minv` diagonal, so a power iteration on the step kernel itself
    converges to that radius. Exact for any order, any material and any boundary layout,
    where `stable_dt` only knows the wave speed and the spacing.

    Args:
        sim: the simulation to measure, whose own `dt` sets the scale of the answer.
        indicator: the design field, which is what a high contrast enters through.
        iterations: power iterations, 60 being ample for three digits.
        safety: fraction of the bound to return.

    Returns:
        the timestep to build `sim` with. A wide stencil over a strong contrast can put
        this far below `stable_dt`, which is the signal to drop `space_order` rather
        than to shrink `dt`.
    """
    mat = sim.build_materials(indicator)
    step = define_step_method(sim, compile_kernels(sim), mat)
    shifts = sim.component_offsets
    if shifts is None:
        shifts = np.zeros((sim.ncomp, sim.ndim))
    slices = [
        (c, *(slice(1, n - 1 - (shifts[c][d] > 0)) for d, n in enumerate(sim.Nx)))
        for c in range(sim.ncomp)
    ]
    field = cp.asarray(
        np.random.default_rng(0).standard_normal((sim.ncomp, *sim.Nx_padded)),
        dtype=sim.dtype,
    )
    zero = cp.zeros_like(field)
    out = cp.zeros_like(field)

    def masked(values):
        kept = cp.zeros_like(values)
        for sl in slices:
            kept[sl] = values[sl]
        return kept

    field = masked(field)
    value = 0.0
    for _ in range(iterations):
        field /= cp.linalg.norm(field)
        out[...] = 0.0
        step(zero, field, out)
        applied = masked(2.0 * field - out)
        value = float(cp.sum(field * applied))
        field = applied
    if value <= 0.0:
        raise ValueError(f"the operator came back non-positive: {value}")
    return safety * sim.dt * float(np.sqrt(4.0 / value))


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


# --------------------------------- staggered lattice ---------------------------------
def component_weights(sim: Simulation, c: int) -> cpt.NDArray:
    """Cell weights of component `c`: halved on the walls of the unstaggered axes."""
    w = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    for d in range(sim.ndim):
        if d == c:
            continue
        for index in (1, sim.Nx[d] - 2):
            wall = [slice(None)] * sim.ndim
            wall[d] = index
            w[tuple(wall)] *= 0.5
    return w


def pair_weights(sim: Simulation, axes: tuple[int, int]) -> cpt.NDArray:
    """Cell weights of a pair point: halved on the walls of the remaining axes."""
    w = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    for d in range(sim.ndim):
        if d in axes:
            continue
        for index in (1, sim.Nx[d] - 2):
            wall = [slice(None)] * sim.ndim
            wall[d] = index
            w[tuple(wall)] *= 0.5
    return w


def point_average(sim: Simulation, field: cpt.NDArray, c: int) -> cpt.NDArray:
    """Arithmetic mean of `field` over the two nodes component `c` sits between."""
    out = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    lo = tuple(slice(0, n - 1) if d == c else slice(0, n) for d, n in enumerate(sim.Nx))
    hi = tuple(slice(1, n) if d == c else slice(0, n) for d, n in enumerate(sim.Nx))
    out[lo] = 0.5 * (field[lo] + field[hi])
    return out


def point_average_adjoint(sim: Simulation, density: cpt.NDArray, c: int) -> cpt.NDArray:
    """Transpose of `point_average`: half of `density` back onto each node it spans."""
    out = 0.5 * density
    to = [slice(None)] * sim.ndim
    fro = [slice(None)] * sim.ndim
    to[c], fro[c] = slice(1, None), slice(0, -1)
    out[tuple(to)] += 0.5 * density[tuple(fro)]
    return out


def pair_average(
    sim: Simulation, field: cpt.NDArray, axes: tuple[int, int]
) -> cpt.NDArray:
    """Harmonic mean of `field` over the four nodes a pair point straddles.

    Harmonic for the reason the cell scheme uses it: it keeps the coefficient single
    valued across a material jump.
    """
    out = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    safe = cp.maximum(field, cp.finfo(sim.dtype).tiny)
    inner = tuple(
        slice(0, n - 1) if d in axes else slice(0, n) for d, n in enumerate(sim.Nx)
    )
    for bits in itertools.product((0, 1), repeat=2):
        shifted = tuple(
            slice(bits[axes.index(d)], n - 1 + bits[axes.index(d)])
            if d in axes
            else slice(0, n)
            for d, n in enumerate(sim.Nx)
        )
        out[inner] += 1.0 / safe[shifted]
    out[inner] = 4.0 / out[inner]
    return out


def pair_average_adjoint(
    sim: Simulation, density: cpt.NDArray, field: cpt.NDArray, axes: tuple[int, int]
) -> cpt.NDArray:
    """Transpose of `pair_average`: d(harmonic mean)/d(node) is `(mean / node)**2 / 4`."""
    safe = cp.maximum(field, cp.finfo(sim.dtype).tiny)
    mean = pair_average(sim, field, axes)
    scattered = density * mean * mean * 0.25
    out = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    for bits in itertools.product((0, 1), repeat=2):
        to = [slice(None)] * sim.ndim
        fro = [slice(None)] * sim.ndim
        for d, b in zip(axes, bits):
            to[d], fro[d] = slice(b, None), slice(0, -b if b else None)
        out[tuple(to)] += scattered[tuple(fro)] / safe[tuple(to)] ** 2
    return out


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

    kernel_path = None  # the forward source this equation compiles, set by the subclass
    sensitivity_path = None  # and the adjoint one
    default_boundary = None  # what `boundary=None` means for this equation

    @property
    def ncomp(self) -> int:
        """Field components per node: 1 for a scalar unknown, `ndim` for a vector one."""
        return 1

    @property
    def component_offsets(self) -> npt.NDArray[np.float64] | None:
        """(ncomp, ndim) grid offsets of each component in units of `dx`, or None for nodal."""
        return None

    @property
    def reach(self) -> int:
        """Nodes one step reads past a point, which a reconstruction strip must cover."""
        return self.space_order // 2

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

    def define_step(self, kernels: cp.RawModule, mat: dict) -> Callable:
        """Closure launching the finite-difference step kernel over (u0, u1, u2)."""
        fd_kernel = kernels.get_function("fd_kernel")
        grid, block = grid_block(self)
        args = [
            None,
            None,
            None,
            *self.step_kernel_args(mat),
            *axis_geometry(self, self.step_factors()),
        ]

        def fd_step(u0, u1, u2):
            args[0], args[1], args[2] = u0, u1, u2
            fd_kernel(grid, block, args)
            return u2

        return fd_step


# ----------------------------------- kernel helpers ----------------------------------
COMMON_PATH = Path(__file__).parent / "kernels" / "common.cuh"


def compile_kernels(sim: Simulation, path: Path | None = None) -> cp.RawModule:
    """Compile `path` for `sim`, defaulting to its own source, stencil table injected."""
    path = sim.kernel_path if path is None else path
    options = ["--use_fast_math", f"-DNDIM={sim.ndim}", *sim.compile_flags]
    if sim.precision == "float32":
        options.append("-DUSE_FLOAT")
    # injected as source, so the module cache keys on the order without a -D flag
    code = preamble(sim.space_order) + COMMON_PATH.read_text() + Path(path).read_text()
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
    """The step closure `sim.define_step` builds, a hook so a scheme may launch several kernels."""
    return sim.define_step(kernels, mat)


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
