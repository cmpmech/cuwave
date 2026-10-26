"""Scalar pressure wave equations, the nodal sibling of the vector elastic schemes.

One unknown per node and a per-axis flux, so the operator is the cheapest of the three
families and needs no component offsets. `PressureWave` holds the nodal `stiff`/`minv`
fields and the adjoint hooks; `ScalarWave` scales both with one indicator gamma, which
is why `minv` is never formed (`derive_inertia`), and `AcousticWave` interpolates
inverse density and inverse bulk modulus between two phases for topology optimization.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
import numpy as np
import numpy.typing as npt

from .boundary import Dirichlet, Neumann, face_mask
from .wave import (
    PML_THREADS,
    Blocked,
    Simulation,
    Source,
    apply_cell_weights,
    axis_geometry,
    compile_kernels,
    flatten_indices,
    grid_block,
    mirror_ghosts,
    sensor_cell_weights,
    slab_grid,
)

KERNEL_PATH = Path(__file__).parent / "kernels" / "scalar.cu"
SENSITIVITY_PATH = Path(__file__).parent / "kernels" / "scalar_sensitivity.cu"
CHUNK_ROWS = 64  # axis-0 rows one streaming block marches
# by stencil radius, candidates (tile, thread rows, steps): the fewest waves of tiles
SMALL_BLOCKS = (
    (((32, 64), 8, 6), ((64, 64), 16, 12)),
    (((32, 64), 8, 4), ((32, 64), 8, 3), ((64, 64), 16, 6)),
)
LARGE_BLOCKS = (((64, 64), 16, 6), None)  # by radius, for fields spilling out of L2
STREAM_ORDER = 6  # below it L1 serves the plain step as well as the stream would
STREAM_BLOCKS = 8  # per multiprocessor the stream needs to fill the device


def _cores(Nx: tuple[int, ...], tile: tuple[int, int], halo: int) -> list[int]:
    """Tiles per axis whose cores, `tile` less `halo` each side, cover the interior."""
    return [-(-(n - 2) // (t - 2 * halo)) for n, t in zip(Nx, tile)]


def _tile_lists(
    points: npt.NDArray[np.int64],
    owners: npt.NDArray[np.int64],
    tile: tuple[int, int],
    core: list[int],
    halo: int,
    cores: list[int],
    covering: bool,
) -> list[cpt.NDArray]:
    """The (2, num) `points` each blocked tile holds: CSR row starts, entries `local | j << 16`.

    With `covering` every tile whose nodes, halo included, take a point holds it, else
    the one tile whose core holds its `owners` node.
    """
    j = np.arange(points.shape[1])
    hits = []
    if covering:
        # tiles starting at 1 + b C - halo hold rows up to tile - 1 past their start
        low = [(points[d] - 1 + halo - tile[d]) // core[d] + 1 for d in (0, 1)]
        high = [(points[d] - 1 + halo) // core[d] for d in (0, 1)]
        spans = [-(-tile[d] // core[d]) for d in (0, 1)]
        for o0, o1 in itertools.product(range(spans[0]), range(spans[1])):
            b = (low[0] + o0, low[1] + o1)
            ok = (b[0] <= high[0]) & (b[1] <= high[1]) & (b[0] >= 0) & (b[1] >= 0)
            ok &= (b[0] < cores[0]) & (b[1] < cores[1])
            hits.append((b[0][ok], b[1][ok], points[0][ok], points[1][ok], j[ok]))
    else:
        b = [(owners[d] - 1) // core[d] for d in (0, 1)]
        hits.append((b[0], b[1], points[0], points[1], j))

    # the entries sorted by tile, each row's start found by bisection
    b0, b1, a0, a1, index = (np.concatenate(h) for h in zip(*hits))
    local = (a0 - 1 - b0 * core[0] + halo) * tile[1] + a1 - 1 - b1 * core[1] + halo
    ids = b0 * cores[1] + b1
    order = np.argsort(ids, kind="stable")
    entries = (local | index << 16)[order].astype(np.uint32)
    starts = np.searchsorted(ids[order], np.arange(cores[0] * cores[1] + 1))
    return [cp.asarray(starts, dtype=cp.int32), cp.asarray(entries)]


def split_grid_block(sim: Simulation) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """`grid_block` with the fast axis split: every row's near-wall columns packed, then the deep ones."""
    grid, block = grid_block(sim)
    radius = sim.space_order // 2
    reach = 2 * radius if radius > 1 else 3  # WALL_REACH in scalar_sensitivity.cu
    walls = -(-2 * (reach - 1) // block[0])
    deep = -(-max(sim.Nx[-1] - 2 * reach, 0) // block[0])
    return (walls + deep, *grid[1:]), block


# -------------------------------- discretization setup -------------------------------
@dataclass
class PressureWave(Simulation):
    """Scalar wave equation base: nodal `stiff`/`minv` material fields, optional damping."""

    kernel_path = KERNEL_PATH
    sensitivity_path = SENSITIVITY_PATH
    default_boundary = Neumann

    derive_inertia = False  # set where m == k (rho scaling): minv derived from stiff
    accepts_domain = True
    accepts_pml = True

    @property
    def streams(self) -> bool:
        """Whether the step streams along axis 0: a wide unmasked 2D stencil on a full grid."""
        # in 3D, or below STREAM_ORDER, L1 serves the plain step as well
        if self.ndim != 2 or self.masked or self.space_order < STREAM_ORDER:
            return False
        tiles = -(-self.Nx_padded[1] // self.threads[-1])
        chunks = -(-(self.Nx[0] - 2) // CHUNK_ROWS)
        sms = cp.cuda.Device().attributes["MultiProcessorCount"]
        return tiles * chunks >= STREAM_BLOCKS * sms

    @property
    def compile_flags(self) -> tuple[str, ...]:
        """The base flags, plus the streaming block where the step streams."""
        flags = super().compile_flags
        if not self.streams:
            return flags
        return flags + ("-DUSE_STREAM", f"-DSTREAM_TX={self.threads[-1]}")

    def define_step(self, kernels: cp.RawModule, mat: dict) -> Callable:
        """Closure launching the step, streamed along axis 0 unless a mask tiles it, then the pml."""
        if self.streams:
            step = self.define_streamed_step(kernels, mat)
        else:
            step = super().define_step(kernels, mat)
        if self.pml is None:
            return step
        pml_step = self.define_pml(kernels, mat)

        def fd_step(u0, u1, u2):
            step(u0, u1, u2)
            return pml_step(u1, u2)

        fd_step.memory = pml_step.memory
        return fd_step

    def define_streamed_step(self, kernels: cp.RawModule, mat: dict) -> Callable:
        """Closure launching the step streamed along axis 0, a block marching rows of it."""
        fd_kernel = kernels.get_function("fd_kernel")
        # a block marches CHUNK_ROWS of axis 0 under its tile of axis 1
        block = (self.threads[-1],)
        grid = (-(-self.Nx_padded[1] // block[0]), -(-(self.Nx[0] - 2) // CHUNK_ROWS))
        args = [None, None, None, *self.step_kernel_args(mat), np.int32(CHUNK_ROWS)]
        args += axis_geometry(self, self.step_factors())

        def fd_step(u0, u1, u2):
            args[0], args[1], args[2] = u0, u1, u2
            fd_kernel(grid, block, args)
            return u2

        return fd_step

    def define_adjoint_step(
        self, kernels: cp.RawModule, sens_kernels: cp.RawModule, mat: dict
    ) -> Callable:
        """Closure stepping the adjoint by the exact transpose of the step's operator."""
        # a mask grades its cells per node, which the transposed kernel does not read
        if self.masked:
            return self.define_step(kernels, mat)
        adjoint_kernel = sens_kernels.get_function("adjoint_kernel")
        grid, block = split_grid_block(self)
        args = [None, None, None, *self.step_kernel_args(mat)]
        args += [face_mask(self, Dirichlet), *axis_geometry(self, self.step_factors())]

        def adjoint_step(l0, l1, l2):
            args[0], args[1], args[2] = l0, l1, l2
            adjoint_kernel(grid, block, args)
            return l2

        if self.pml is None:
            return adjoint_step
        pml_step = self.define_adjoint_pml(sens_kernels, mat)

        def pml_adjoint_step(l0, l1, l2):
            adjoint_step(l0, l1, l2)
            return pml_step(l1, l2)

        pml_adjoint_step.memory = pml_step.memory
        return pml_adjoint_step

    def define_pml(self, kernels: cp.RawModule, mat: dict) -> Callable:
        """Closure adding the pml memory terms of `u1` into `u2`, one launch per lined axis."""
        pml_kernel = kernels.get_function("pml_kernel")
        factors = self.step_factors()
        launches, memory = [], []
        for d, lo, hi, profile, plane in self.pml_slabs():
            # the three memory variables per slab node: plus cell, minus cell, divergence
            aux = cp.zeros((3, (lo + hi) * plane), dtype=self.dtype)
            memory.append(aux)
            args = [None, None, *self.step_kernel_args(mat), aux, profile]
            args += [np.int32(d), np.int32(lo), np.int32(hi), factors[d]]
            launches.append((slab_grid(lo, hi, plane), args + axis_geometry(self)))

        def pml_step(u1, u2):
            for grid, args in launches:
                args[0], args[1] = u1, u2
                pml_kernel(grid, (PML_THREADS,), args)
            return u2

        # the state the step carries besides the field, for a checkpoint to copy
        pml_step.memory = memory
        return pml_step

    def define_adjoint_pml(
        self, kernels: cp.RawModule, mat: dict, grads: dict | None = None
    ) -> Callable:
        """Closure adding the transposed pml terms into `l2` from `l1`, and to `grads` if fused."""
        aux_kernel = kernels.get_function("pml_aux_kernel")
        adjoint_kernel = kernels.get_function("pml_adjoint_kernel")
        R = self.space_order // 2
        factors = self.step_factors()
        fused = grads is not None
        launches, memory = [], []
        for d, lo, hi, profile, plane in self.pml_slabs():
            eta = cp.zeros((3, (lo + hi) * plane), dtype=self.dtype)
            memory.append(eta)
            slab = [np.int32(d), np.int32(lo), np.int32(hi), factors[d]]
            aux_args = [None, mat["stiff"], eta, profile, *slab, *axis_geometry(self)]
            args = [None, grads["mass"] if fused else None, None]
            args += [self.dtype(self.adjoint_mass_factor()), np.int32(fused)]
            args += [*self.step_kernel_args(mat), eta, profile, *slab]
            args += [face_mask(self, Dirichlet), *axis_geometry(self)]
            # the transposed taps land up to R nodes past the layer front
            grids = (slab_grid(lo, hi, plane), slab_grid(lo, hi, plane, R))
            launches.append((grids, aux_args, args))

        def adjoint_pml_step(l1, l2, u1=None):
            for (aux_grid, grid), aux_args, args in launches:
                aux_args[0] = l1
                aux_kernel(aux_grid, (PML_THREADS,), aux_args)
                # unfused, the mass slots are never read, so l2 stands in for them
                args[0], args[2] = l2, l2 if u1 is None else u1
                if not fused:
                    args[1] = l2
                adjoint_kernel(grid, (PML_THREADS,), args)
            return l2

        # the state the adjoint carries besides the field, for a subdomain to exchange
        adjoint_pml_step.memory = memory
        return adjoint_pml_step

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

    def define_blocked_step(
        self, mat: dict, source: Source, sensors: cpt.NDArray[cp.int32] | None
    ) -> Blocked | None:
        """The 2D step advancing several steps a launch, excitation and record included, else None.

        Each thread block loads a tile with a halo of as many stencil radii as it takes
        steps, steps it in shared memory, refills its ghosts there and writes back its
        core: bitwise the step, excitation, refill and record launched one by one. Taken
        where launches or DRAM bound the plain step (`SMALL_BLOCKS`, `LARGE_BLOCKS`), else
        None, as in 3D, in float64, above order 4, with a pml, a domain mask, a condition
        other than Neumann or Dirichlet, a source off the interior, or two sources or
        sensors on one node.
        """
        conditions = {c for pair in self.boundary for c in pair}
        radius = self.space_order // 2
        if (
            self.ndim != 2
            or self.precision != "float32"
            or radius > len(SMALL_BLOCKS)
            or self.pml is not None
            or self.masked
            or not conditions <= {Neumann, Dirichlet}
        ):
            return None
        # launches bound a grid of two waves of threads, DRAM one whose u twice,
        # stiffness and inertia spill out of L2; the halo's recomputation costs elsewhere
        device = cp.cuda.Device().attributes
        resident = device["MultiProcessorCount"] * device["MaxThreadsPerMultiProcessor"]
        nodes = math.prod(self.Nx)
        if nodes <= 2 * resident:
            # one tile a multiprocessor at a time: the candidate with the fewest waves
            sms = device["MultiProcessorCount"]
            block = min(
                SMALL_BLOCKS[radius - 1],
                key=lambda b: (
                    -(-math.prod(_cores(self.Nx, b[0], b[2] * radius)) // sms)
                ),
            )
        elif 16 * nodes > device["L2CacheSize"]:
            block = LARGE_BLOCKS[radius - 1]
        else:
            block = None
        if block is None:
            return None

        # the tiles: a core and a halo as wide as the steps reach
        tile, rows, steps = block
        halo = steps * radius
        core = [t - 2 * halo for t in tile]
        cores = _cores(self.Nx, tile, halo)

        # sources in the interior, no node holding two sources or two sensors
        sources = flatten_indices(self, source.position)
        targets = flatten_indices(self, sensors) if sensors is not None else sources[:0]
        flat = [index.get() for index in (sources, targets)]
        points = [np.stack(np.divmod(index, self.Nx_padded[1])) for index in flat]
        upper = np.asarray(self.Nx)[:, None] - 2
        if not ((points[0] >= 1) & (points[0] <= upper)).all():
            return None
        for index in flat:
            if len(index) >= 2**16 or len(np.unique(index)) != len(index):
                return None
        # a sensor recorded by the tile whose core holds the node a ghost clamps to
        clamped = np.clip(points[1], 1, upper)
        lists = [
            *_tile_lists(points[0], points[0], tile, core, halo, cores, covering=True),
            *_tile_lists(points[1], clamped, tile, core, halo, cores, covering=False),
        ]

        # the kernel and its arguments, the pointers swapped per launch
        flags = (f"-DSTEPS={steps}", f"-DTILE0={tile[0]}", f"-DTILE1={tile[1]}")
        kernels = compile_kernels(self, flags=(*flags, f"-DROWS={rows}"))
        kernel = kernels.get_function("blocked_kernel")
        # u twice, each in a guard ring a radius wide
        shared = 2 * (tile[0] + 2 * radius) * (tile[1] + 2 * radius) * 4
        kernel.max_dynamic_shared_size_bytes = shared
        weights = self.excitation_weights(mat, sources)
        args = [None, None, None, None, *self.step_kernel_args(mat), None, weights]
        at_signal = len(args) - 2
        args += [np.int32(len(sources)), None, np.int32(len(targets)), *lists, None]
        at_record, at_t = len(args) - 7, len(args) - 1
        args += [face_mask(self, Neumann), face_mask(self, Dirichlet)]
        args += axis_geometry(self, self.step_factors())
        dummy = cp.zeros(1, dtype=self.dtype)

        def blocked_step(U, V, t, signal, record=None):
            # U[t % 2] holds u^(t - 1) and U[1 - t % 2] u^t, as the plain rotation does
            s = steps
            args[0], args[1] = U[t % 2], U[1 - t % 2]
            args[2], args[3] = V[(t + s) % 2], V[(t + s - 1) % 2]
            args[at_signal] = signal
            args[at_record] = dummy if record is None else record
            args[at_t] = np.int32(t)
            kernel(tuple(cores[::-1]), (tile[1], rows), args, shared_mem=shared)

        return Blocked(steps, blocked_step)

    def step_kernel_args(self, mat: dict) -> tuple:
        """Material and damping arguments for the finite-difference step kernel."""
        minv = mat["stiff"] if self.derive_inertia else mat["minv"]
        args = (mat["stiff"], minv, np.int32(self.derive_inertia))
        if self.damping is not None:
            args += (mat["damping"], self.dtype(self.dt))
        return args

    gradient_names = ("mass", "stiff")  # the fields the adjoint differentiates
    fuses_adjoint = True  # one kernel steps the adjoint and accumulates both gradients

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
        grid, block = split_grid_block(self)
        # the operator without the dt^2 the step folds into it: L, not dt^2 L
        factors = [self.dtype(float(f) / self.dt**2) for f in self.step_factors()]
        args = [grads["mass"], grads["stiff"], None, None, None, None, mat["stiff"]]
        args += [self.dtype(1.0 / self.dt**2), *axis_geometry(self, factors)]

        def gradient_step(u0, u1, u2, l1):
            args[2], args[3], args[4], args[5] = u0, u1, u2, l1
            gradient_kernel(grid, block, args)

        return gradient_step

    def define_adjoint_gradient(
        self, kernels: cp.RawModule, mat: dict, grads: dict
    ) -> Callable:
        """Closure stepping the adjoint `l0` over `l1`, accumulating both gradients of `l1`."""
        adjoint_gradient_kernel = kernels.get_function("adjoint_gradient_kernel")
        grid, block = split_grid_block(self)
        args = [None, None, grads["mass"], grads["stiff"], None]
        args += [*self.step_kernel_args(mat), face_mask(self, Dirichlet)]
        args += [self.dtype(self.adjoint_mass_factor())]
        args += axis_geometry(self, self.step_factors())

        def adjoint_gradient_step(l0, l1, u1):
            args[0], args[1], args[4] = l0, l1, u1
            adjoint_gradient_kernel(grid, block, args)
            return l0

        if self.pml is None:
            return adjoint_gradient_step
        pml_step = self.define_adjoint_pml(kernels, mat, grads)

        def pml_adjoint_gradient_step(l0, l1, u1):
            adjoint_gradient_step(l0, l1, u1)
            return pml_step(l1, l0, u1)

        pml_adjoint_gradient_step.memory = pml_step.memory
        return pml_adjoint_gradient_step

    def define_superposed(
        self, kernels: cp.RawModule, mat: dict, accs: dict, sign: float
    ) -> Callable:
        """Closure stepping (u0, u1, u2), adding `sign` times the stiffness density of this triplet and the mass density of the one before."""
        superposed_kernel = kernels.get_function("superposed_kernel")
        grid, block = grid_block(self)
        factors = self.step_factors()
        # both per-axis factors go as 1 / dx**2, so their ratio is one number
        ratios = [-sign / (2.0 * d) ** 2 / float(f) for d, f in zip(self.dx, factors)]
        if not np.allclose(ratios, ratios[0]):
            raise ValueError(
                f"step factors {factors} are not proportional to 1 / dx**2"
            )
        args = [None, None, None, accs["mass"], accs["stiff"]]
        args += [*self.step_kernel_args(mat), self.dtype(sign / (2.0 * self.dt) ** 2)]
        args += [self.dtype(ratios[0]), *axis_geometry(self, factors)]
        massless = list(args)
        massless[8] = self.dtype(0.0)

        def superposed_step(u0, u1, u2, mass=True):
            launch = args if mass else massless
            launch[0], launch[1], launch[2] = u0, u1, u2
            superposed_kernel(grid, block, launch)
            return u2

        return superposed_step

    def define_frechet(
        self, kernels: cp.RawModule, accs: dict, sign: float, stiffness: bool = True
    ) -> Callable:
        """Closure accumulating both Frechet densities of one field triplet, times `sign`."""
        # sign is fixed per pass, so it is folded into the factors, not recomputed
        frechet_kernel = kernels.get_function("frechet_kernel")
        grid, block = grid_block(self)
        # the stiffness density enters negated, so the epilogue scales both alike
        factors = [self.dtype(-sign * stiffness / (2.0 * d) ** 2) for d in self.dx]
        args = [accs["mass"], accs["stiff"], None, None, None]
        args += [self.dtype(sign / (2.0 * self.dt) ** 2), *axis_geometry(self, factors)]

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
            # the damped update divides by 1 + beta, so the source has to share it
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
