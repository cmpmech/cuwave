"""The grid, the compile-time configuration, and the time loop every equation shares.

`Simulation` holds both and leaves the physics to a subclass in its own module
(`scalar.py`, `elastic.py`, `anisotropic.py`, `maxwell.py`), which names its kernel
sources and supplies the material and factor hooks. The `define_*` factories bind
compiled kernels to one such configuration, and `simulate` loops over the closures they
return.
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

# Voigt row order per dimension as (k, l) strain pairs; the shear rows are PAIRS[d][d:]
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

    The leapfrog is stable while the spectral radius of `dt**2 minv L` stays under 4,
    and `L` is symmetric with `minv` diagonal, so a power iteration on the step kernel
    itself converges to that radius. Exact for any order, any material and any boundary
    layout, where `stable_dt` only knows the wave speed and the spacing.

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
    step = sim.define_step(compile_kernels(sim), mat)
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
def wall_weights(sim: Simulation, axes: tuple[int, ...]) -> cpt.NDArray:
    """Cell weights of a point family staggered along `axes`: halved on the other walls."""
    w = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    for d in set(range(sim.ndim)) - set(axes):
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
    damping: cpt.NDArray | None = None  # nodal field d, None for a lossless operator
    domain: cpt.NDArray | None = None  # nodal mask of the physical domain, None for all
    dirichlet: cpt.NDArray | None = None  # nodal mask held at zero, its cells left open

    accepts_domain = False  # whether the step kernel honours -DUSE_DOMAIN
    fuses_adjoint = False  # whether the gradients ride in the adjoint step kernels

    @property
    def compile_flags(self) -> tuple[str, ...]:
        """`-DUSE_DAMPING` and `-DUSE_DOMAIN` for the fields that are set."""
        if self.masked and not self.accepts_domain:
            raise ValueError(f"{type(self).__name__} steps every node: unset domain")
        flags = ("-DUSE_DAMPING",) if self.damping is not None else ()
        return flags + (("-DUSE_DOMAIN",) if self.masked else ())

    @property
    def masked(self) -> bool:
        """Whether `domain` or `dirichlet` takes nodes out of the stepped set."""
        return self.domain is not None or self.dirichlet is not None

    kernel_path = None  # forward source this equation compiles, set by the subclass
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

    def adjoint_mass_factor(self) -> float:
        """Scale of the inertia gradient density: `1 / dt**2`, the residual's own."""
        return 1.0 / self.dt**2

    def define_step(self, kernels: cp.RawModule, mat: dict) -> Callable:
        """Closure launching the finite-difference step kernel over (u0, u1, u2)."""
        fd_kernel = kernels.get_function("fd_kernel")
        grid, block = grid_block(self)
        args = [None, None, None, *self.step_kernel_args(mat)]
        if self.masked:
            cells = domain_cells(self)
            tiles = domain_tiles(self, cells)
            grid = (tiles.shape[0],)
            args += [cells, tiles]
        args += axis_geometry(self, self.step_factors())

        def fd_step(u0, u1, u2):
            args[0], args[1], args[2] = u0, u1, u2
            fd_kernel(grid, block, args)
            return u2

        return fd_step

    def define_adjoint_step(
        self, kernels: cp.RawModule, sens_kernels: cp.RawModule, mat: dict
    ) -> Callable:
        """Closure stepping the adjoint field: `define_step`, wherever that is its own transpose."""
        return self.define_step(kernels, mat)


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


DEEP = 1 << 30  # the cell code of a node whose stencil never reaches the wall
DIRICHLET = 7  # the radius code of a cell open onto a node held at zero


def stepped_nodes(sim: Simulation) -> cpt.NDArray[cp.bool_]:
    """The nodes the step updates: `domain` (all by default) less `dirichlet`."""
    stepped = cp.ones(sim.Nx_padded, dtype=cp.bool_)
    if sim.domain is not None:
        stepped &= sim.domain
    if sim.dirichlet is not None:
        stepped &= ~sim.dirichlet
    return stepped


def domain_cells(sim: Simulation) -> cpt.NDArray[cp.int32]:
    """Per stepped node, the radii of its two cells along each axis, packed.

    A cell's stencil may only reach stepped nodes, so its radius is capped by the run
    of them on either side, which is zero for the cell leaving them: that is the
    zero-flux wall, unless the node across is `dirichlet`, which leaves the cell open
    onto a zero at its midpoint (`DIRICHLET`), where the Neumann wall sits too. The code is 0 for a node not stepped,
    `DEEP` where every radius is the full one, and otherwise (rp | rm << 3) << 6 * axis.
    """
    R = sim.space_order // 2
    if R >= DIRICHLET:
        raise ValueError(
            f"space_order {sim.space_order} overflows the 3-bit cell radius"
        )
    interior = tuple(slice(1, n - 1) for n in sim.Nx)
    # ghost nodes count as stepped, so the closure at the box walls is unchanged
    inside = cp.ones(sim.Nx_padded, dtype=cp.bool_)
    inside[interior] = stepped_nodes(sim)[interior]
    held = cp.zeros(sim.Nx_padded, dtype=cp.bool_)
    if sim.dirichlet is not None:
        held[interior] = sim.dirichlet[interior]

    # `mask` k nodes along axis d, `fill` past the array
    def shifted(mask, d, k, fill):
        window = [slice(R, R + n) for n in sim.Nx_padded]
        window[d] = slice(R + k, R + k + sim.Nx_padded[d])
        return cp.pad(mask, R, constant_values=fill)[tuple(window)]

    code = cp.zeros(sim.Nx_padded, dtype=cp.int32)
    deep = inside.copy()
    for d in range(sim.ndim):
        shape = [1] * sim.ndim
        shape[d] = -1
        a = cp.arange(sim.Nx_padded[d]).reshape(shape)
        wall = cp.minimum(R, cp.minimum(a, sim.Nx[d] - 1 - a))  # CLOSURE

        def run(sign):
            count, on = cp.zeros(sim.Nx_padded, cp.int32), cp.ones_like(inside)
            for k in range(1, R + 1):
                on &= shifted(inside, d, sign * k, True)
                count += on & (k <= wall)
            return count

        up, down = run(1), run(-1)
        rp = cp.minimum(wall, cp.minimum(1 + down, up))
        rm = cp.minimum(wall, cp.minimum(down, 1 + up))
        rp = cp.where((rp == 0) & shifted(held, d, 1, False), DIRICHLET, rp)
        rm = cp.where((rm == 0) & shifted(held, d, -1, False), DIRICHLET, rm)
        code |= (rp | rm << 3) << (6 * d)
        deep &= (rp == wall) & (rm == wall)
    code = cp.where(deep, DEEP, code)
    return cp.where(inside, code, 0).astype(cp.int32)


def domain_tiles(
    sim: Simulation, cells: cpt.NDArray[cp.int32]
) -> cpt.NDArray[cp.int32]:
    """The tiles holding any stepped node, in C order, one packed int each.

    A block index takes 10 bits an axis with the fastest lowest, and bit 30 marks a
    tile holding a node that is not `DEEP`, the only tiles that read `cells`.
    """
    grid = [-(-n // t) for n, t in zip(sim.Nx_padded, sim.threads)]
    if max(grid) > 1024:
        raise ValueError(f"{grid} blocks per axis overflow the 10-bit tile packing")
    interior = tuple(slice(1, n - 1) for n in sim.Nx)
    within = tuple(range(1, 2 * sim.ndim, 2))

    # per tile, with the nodes the kernel never steps filled so they decide nothing
    def reduce(fill, test):
        tiled = cp.full([g * t for g, t in zip(grid, sim.threads)], fill, cp.int32)
        tiled[interior] = cells[interior]
        tiled = tiled.reshape([s for g, t in zip(grid, sim.threads) for s in (g, t)])
        return test(tiled).any(axis=within)

    occupied = reduce(0, lambda t: t != 0)
    wall = reduce(DEEP, lambda t: t != DEEP)
    blocks = cp.argwhere(occupied).astype(cp.int32)
    tiles = wall[tuple(blocks.T)].astype(cp.int32) << 30
    for d in range(sim.ndim):
        tiles |= blocks[:, d] << (10 * (sim.ndim - 1 - d))
    return tiles


def axis_geometry(sim: Simulation, factors: list | None = None) -> list:
    """Interleave `factors` with axis extents and strides, in the layout the kernels expect."""
    # kernel args after the material arrays: f0, N0, [f1, N1, s0], [f2, N2, s1]
    geom = []
    for d in range(sim.ndim):
        geom += [] if factors is None else [factors[d]]
        geom += [sim.Nx[d]] + ([sim.strides[d - 1]] if d else [])
    return geom


# ----------------------------------- time marching -----------------------------------
CHUNK = 24  # steps per captured graph, a multiple of every rotation period (2, 3)


@dataclass
class Window:
    """A record `march` stages for a graph: `step` reads or writes it at row t + offset.

    An input loads the `halo` rows past each chunk as well; a store window writes its
    rows `halo ..` back and, if it also loads, loads only the `halo` rows it starts from.
    """

    record: cpt.NDArray
    halo: int = 0
    load: bool = True
    store: bool = False

    def __post_init__(self) -> None:
        """An output only stores, so `store=True` alone clears `load`."""
        self.load = self.load and (not self.store or self.halo > 0)


def march(N: int, step: Callable, windows: list[Window] = ()) -> None:
    """Call `step(t, *records)` for t in range(N), replaying chunks as a captured graph.

    Per-step host work is the launch itself, which on a small grid outweighs the kernel,
    so `CHUNK` steps are captured once and replayed. `step` may launch kernels only,
    rotate its slots by t modulo a divisor of `CHUNK`, and touch each window's record
    at rows t .. t + halo. A graph addresses fixed memory, so the windows are staged
    through buffers of `CHUNK + halo` rows, and only while they fit in L2 (there the
    staging copy is free); past that the kernels outweigh the launch anyway.

    Args:
        N: the number of steps.
        step: launches step t, taking the windows' records in order.
        windows: the records `step` reads or writes per step, each a `Window`.
    """
    shapes = [(CHUNK + w.halo, *w.record.shape[1:]) for w in windows]
    nbytes = sum(int(np.prod(s)) * w.record.itemsize for s, w in zip(shapes, windows))
    if N < 2 * CHUNK or nbytes > cp.cuda.Device().attributes["L2CacheSize"]:
        for t in range(N):
            step(t, *(w.record for w in windows))
        return

    staging = [cp.empty(s, dtype=w.record.dtype) for s, w in zip(shapes, windows)]
    capture = cp.cuda.Stream()
    with capture:
        capture.begin_capture()
        for k in range(CHUNK):
            step(k, *staging)
        graph = capture.end_capture()
    for start in range(0, N, CHUNK):
        count = min(CHUNK, N - start)
        for w, buf in zip(windows, staging):
            if w.load:
                rows = w.halo if w.store else count + w.halo
                buf[:rows] = w.record[start : start + rows]
        if count == CHUNK:
            graph.launch()
        else:
            for k in range(count):
                step(k, *staging)
        for w, buf in zip(windows, staging):
            if w.store:
                w.record[start + w.halo : start + w.halo + count] = buf[
                    w.halo : w.halo + count
                ]


# -------------------------------- simulation functions -------------------------------
def _define_transfer(
    sim: Simulation,
    name: str,
    position: cpt.NDArray[cp.int32],
    kernels: cp.RawModule,
    *extra: cpt.NDArray,
) -> Callable:
    """Closure launching transfer kernel `name` between `u` and row t of a record."""
    if sim.masked:
        inside = stepped_nodes(sim)[tuple(grid_rows(sim, position))]
        if not bool(inside.all()):
            raise ValueError(f"{name} reaches a node that is never stepped")
    kernel = kernels.get_function(name)
    threads = 256
    num = position.shape[1]
    blocks = ((num + threads - 1) // threads,)
    # the whole (N, num) record plus a row offset, not a row view
    args = [None, None, np.int32(0), flatten_indices(sim, position), np.int32(num)]
    args += extra

    def transfer_step(u, record, t_index):
        args[0], args[1] = u, record
        args[2] = np.int32(t_index * num)
        kernel(blocks, (threads,), args)
        return u

    return transfer_step


def define_excitation(
    sim: Simulation,
    position: cpt.NDArray[cp.int32],
    kernels: cp.RawModule,
    mat: dict,
) -> Callable:
    """Closure adding row `t_index` of an (N, num_sources) signal into `u` at `position`."""
    weight = sim.excitation_weights(mat, flatten_indices(sim, position))
    return _define_transfer(sim, "excitation_kernel", position, kernels, weight)


def define_adjoint_excitation(
    sim: Simulation,
    sensors: cpt.NDArray[cp.int32],
    kernels: cp.RawModule,
    mat: dict,
    grads: dict,
) -> Callable:
    """Closure adding row `t_index` of the adjoint signal into `l2`, and its share of the inertia gradient against `u1`."""
    lin_index = flatten_indices(sim, sensors)
    kernel = kernels.get_function("adjoint_excitation_kernel")
    threads = 256
    num = sensors.shape[1]
    blocks = ((num + threads - 1) // threads,)
    args = [None, None, np.int32(0), lin_index, np.int32(num)]
    args += [sim.excitation_weights(mat, lin_index), grads["mass"], None]
    args.append(sim.dtype(sim.adjoint_mass_factor()))

    def adjoint_excitation_step(l2, signal, u1, t_index):
        args[0], args[1], args[7] = l2, signal, u1
        args[2] = np.int32(t_index * num)
        kernel(blocks, (threads,), args)
        return l2

    return adjoint_excitation_step


def define_get_signal(
    sim: Simulation, sensors: cpt.NDArray[cp.int32], kernels: cp.RawModule
) -> Callable:
    """Closure writing row `t_index` of the (N, num_sensors) record `um` from `u`."""
    return _define_transfer(sim, "get_signal_kernel", sensors, kernels)


def define_set_signal(
    sim: Simulation, sensors: cpt.NDArray[cp.int32], kernels: cp.RawModule
) -> Callable:
    """Closure writing `u` at `sensors` back from row `t_index`, restoring rather than adding."""
    return _define_transfer(sim, "set_signal_kernel", sensors, kernels)


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

    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    fd_step = sim.define_step(kernels, mat)
    bc_step = define_boundary(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    windows = [Window(source.signal)]
    if sensors is not None:
        get_signal = define_get_signal(sim, sensors, kernels)
        um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)
        windows.append(Window(um, store=True))
    interior = (Ellipsis, *(slice(0, n) for n in sim.Nx))

    # U[t % 2] takes u^t over u^(t - 2), so the rotation is a function of t alone
    def step(t, signal, um=None):
        u = fd_step(U[t % 2], U[1 - t % 2], U[t % 2])
        excitation_step(u, signal, t)
        bc_step(u)
        if um is not None:
            get_signal(u, um, t)

    if record_every is None:
        march(sim.N, step, windows)
    else:
        snapshots = []
        for t in range(sim.N):
            step(t, *(w.record for w in windows))
            if t % record_every == 0:
                snapshots.append(U[t % 2][interior].get())

    out = (U[(sim.N - 1) % 2][interior],)
    if sensors is not None:
        out += (um,)
    if record_every is not None:
        out += (np.stack(snapshots),)
    return out if len(out) > 1 else out[0]
