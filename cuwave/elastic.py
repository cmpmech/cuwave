"""Isotropic elasticity on a staggered grid, a sibling of `PressureWave` on the same march.

Component `c` lives half a node up its own axis, so every strain lands on a natural
point: the normal strains on the nodes, the shear `(k, l)` half a node up both of its
axes. Each is a pure per-axis staggered difference, which is what carries the cross
terms a per-axis flux cannot and keeps the cost linear in the stencil radius, where the
cell gather of `AnisotropicElasticWave` pays `(2r)**(2 ndim)`. The operator is
`-B^T C B` with the material sampled on the stress points, so it stays the exact
transpose at every order.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
import numpy as np
import numpy.typing as npt

from .boundary import Clamped, Traction, face_mask, faces_with
from .wave import (
    PAIRS,
    Simulation,
    apply_cell_weights,
    axis_geometry,
    compile_kernels,
    grid_block,
    pair_average_adjoint,
    point_average_adjoint,
    wall_weights,
)

KERNEL_PATH = Path(__file__).parent / "kernels" / "elastic.cu"
SENSITIVITY_PATH = Path(__file__).parent / "kernels" / "elastic_sensitivity.cu"
FUSED_TILE = ((16, 32), (4, 8, 32))  # in 2D, 3D: the fused step's tile, a thread each


# -------------------------------------- helpers --------------------------------------
def voigt(ndim: int, lame: float, shear: float, plane: str = "strain") -> npt.NDArray:
    """Isotropic Voigt stiffness matrix, `plane` selecting strain or stress in 2D."""
    if ndim == 1:
        return np.array([[lame + 2.0 * shear]])
    if ndim == 2:
        lam = lame if plane == "strain" else 2.0 * lame * shear / (lame + 2.0 * shear)
        return np.array(
            [
                [lam + 2.0 * shear, lam, 0.0],
                [lam, lam + 2.0 * shear, 0.0],
                [0.0, 0.0, shear],
            ]
        )
    C = np.full((6, 6), 0.0)
    C[:3, :3] = lame
    for i in range(3):
        C[i, i] = lame + 2.0 * shear
        C[3 + i, 3 + i] = shear
    return C


# ------------------------------- discretization setup --------------------------------
@dataclass
class ElasticWave(Simulation):
    """Staggered isotropic elasticity, parametrized by a density-scaling indicator gamma.

    Both wave speeds are held fixed and gamma scales the density, so
    `C = gamma * rho0 * C` scales inertia and stiffness alike and the stable timestep
    does not move with the design. This is the parametrization ultrasonic full waveform
    inversion reaches for, since a void is a density contrast at unchanged speeds.
    """

    density: float = None  # background density rho0
    wavespeed_p: float = None  # pressure wave speed
    wavespeed_s: float = None  # shear wave speed
    plane: str = "strain"  # "strain" or "stress", 2D only

    kernel_path = KERNEL_PATH
    sensitivity_path = SENSITIVITY_PATH
    default_boundary = Traction
    gradient_names = ("mass", "stiff")
    accepts_pml = True
    pml_shifted = True
    pml_walls = (Clamped,)  # a traction wall behind the layer holds a growing surface wave

    @property
    def ncomp(self) -> int:
        """One displacement component per axis."""
        return self.ndim

    @property
    def component_offsets(self) -> npt.NDArray[np.float64]:
        """Component `c` sits half a node up axis `c`: the staggering itself."""
        return 0.5 * np.eye(self.ndim)

    @property
    def nvoigt(self) -> int:
        """Stress components: `ndim` normal ones plus one per axis pair."""
        return len(PAIRS[self.ndim])

    @property
    def npairs(self) -> int:
        """Shear components, one per axis pair."""
        return self.nvoigt - self.ndim

    def __post_init__(self) -> None:
        """Validate the material, the 2D plane assumption and the faces, then derive the grid."""
        super().__post_init__()
        if None in (self.density, self.wavespeed_p, self.wavespeed_s):
            raise ValueError("ElasticWave requires density, wavespeed_p, wavespeed_s")
        if self.plane not in ("strain", "stress"):
            raise ValueError(f"plane must be strain or stress: {self.plane}")
        if self.ndim == 3 and self.plane != "strain":
            raise ValueError("plane stress is a 2D reduction, not a 3D one")
        if self.wavespeed_s >= self.wavespeed_p:
            raise ValueError(
                f"wavespeed_s must be below wavespeed_p: "
                f"{self.wavespeed_s} >= {self.wavespeed_p}"
            )
        for pair in self.boundary:
            for condition in pair:
                if condition not in (Traction, Clamped):
                    raise ValueError(f"elastic faces are Traction or Clamped: {pair}")

    @property
    def compile_flags(self) -> tuple[str, ...]:
        """The base flags, plus `-DUSE_PML` folding the layer into both step kernels."""
        return super().compile_flags + (("-DUSE_PML",) if self.pml is not None else ())

    @property
    def radius(self) -> int:
        """Half the stencil width, so `space_order` is twice it."""
        return self.space_order // 2

    @property
    def reach(self) -> int:
        """The strain reach compounds with the divergence one: `2 radius - 1` nodes."""
        return 2 * self.radius - 1

    @property
    def pml_margin(self) -> int:
        """Undamped layer nodes before the front: a stress the step corrects is read this far."""
        return self.radius - 1

    @property
    def lame(self) -> float:
        """First Lame parameter `rho0 * (c_p**2 - 2 c_s**2)`, reduced under plane stress."""
        lam = self.density * (self.wavespeed_p**2 - 2.0 * self.wavespeed_s**2)
        if self.plane == "stress":
            return 2.0 * lam * self.shear / (lam + 2.0 * self.shear)
        return lam

    @property
    def shear(self) -> float:
        """Second Lame parameter `rho0 * c_s**2`."""
        return self.density * self.wavespeed_s**2

    def inverse_inertia(self, indicator: cpt.NDArray) -> cpt.NDArray:
        """Nodal `1 / (gamma rho0 W)`, what a sponge scales its damping by."""
        mass = (
            indicator
            * self.density
            * apply_cell_weights(self, cp.ones(self.Nx_padded, dtype=self.dtype))
        )
        return 1.0 / cp.maximum(mass, cp.finfo(self.dtype).tiny)

    def build_materials(self, indicator: cpt.NDArray) -> dict:
        """The design field alone: the kernels average it onto the stress and component points."""
        mat = {"gamma": cp.ascontiguousarray(indicator, dtype=self.dtype)}
        if self.damping is not None:
            mat["damping"] = self.damping
        return mat

    def point_inverse_inertia(
        self, gamma: cpt.NDArray, lin_index: cpt.NDArray[cp.int32]
    ) -> cpt.NDArray:
        """`1 / (rho0 W mean gamma)` on the component points `lin_index`, as `point_minv` forms it."""
        component, node = lin_index // self.comp_stride, lin_index % self.comp_stride
        rows = [node // s % n for s, n in zip(self.strides, self.Nx_padded)]
        step = cp.asarray(self.strides, dtype=lin_index.dtype)[component]
        flat = gamma.ravel()
        mean = 0.5 * (flat[node] + flat[node + step])
        weight = cp.ones(lin_index.shape, dtype=self.dtype)
        for d, a in enumerate(rows):
            for index in (1, self.Nx[d] - 2):
                weight = cp.where((a == index) & (component != d), 0.5 * weight, weight)
        mass = self.density * weight * mean
        minv = 1.0 / cp.maximum(mass, cp.finfo(self.dtype).tiny)
        for face in faces_with(self, Clamped):
            d = face // 2
            wall = 1 if face % 2 == 0 else self.Nx[d] - 2
            minv = cp.where((rows[d] == wall) & (component != d), 0.0, minv)
        return minv.astype(self.dtype)

    def define_step(self, kernels: cp.RawModule, mat: dict) -> Callable:
        """Closure launching the stress kernel and then the update over (u0, u1, u2)."""
        return self._define_pair(kernels, mat, adjoint=False)

    def define_adjoint_step(
        self, kernels: cp.RawModule, sens_kernels: cp.RawModule, mat: dict
    ) -> Callable:
        """Closure stepping the adjoint: the step itself, unless a pml breaks its symmetry."""
        return self._define_pair(kernels, mat, adjoint=self.pml is not None)

    def _define_pair(self, kernels: cp.RawModule, mat: dict, adjoint: bool) -> Callable:
        """The stress and update launches, a pml folded into both; fused past L2 without one."""
        # one launch where DRAM bounds the pair (u three times, the stresses, the
        # design) and the ring at most doubles the stresses formed
        device = cp.cuda.Device().attributes
        nodes = math.prod(self.Nx)
        fields = (3 * self.ndim + self.nvoigt + 1) * nodes * self.dtype().itemsize
        tile = FUSED_TILE[self.ndim - 2] if self.ndim >= 2 else ()
        ring = math.prod(t + self.space_order for t in tile)
        if (
            self.ndim >= 2
            and self.pml is None
            and fields > device["L2CacheSize"]
            and ring <= 2 * math.prod(tile)
            and self._fused_shared() <= device["MaxSharedMemoryPerBlockOptin"]
        ):
            return self._define_fused(mat)

        # the pair, a pml folded into both
        grid, block = grid_block(self)
        stress_kernel = kernels.get_function("stress_kernel")
        fd_kernel = kernels.get_function("fd_kernel")
        _, sargs, uargs = self.step_args(mat)
        memory = []
        if self.pml is not None:
            fold, memory = self._pml_fold(adjoint)
            sargs += fold
            uargs += fold

        def fd_step(u0, u1, u2):
            sargs[0] = u1
            stress_kernel(grid, block, sargs)
            uargs[0], uargs[1], uargs[2] = u0, u1, u2
            fd_kernel(grid, block, uargs)
            if adjoint:  # the strain memory swaps its two row blocks
                sargs[-1] = uargs[-1] = np.int32(3 - sargs[-1])
            return u2

        fd_step.memory = memory
        return fd_step

    def _define_fused(self, mat: dict) -> Callable:
        """The step as one launch, each block's stresses in shared memory, ring included."""
        tile = FUSED_TILE[self.ndim - 2]
        flags = tuple(f"-DTILE{d}={t}" for d, t in enumerate(tile))
        kernel = compile_kernels(self, flags=flags).get_function("fused_kernel")
        grid = tuple(-(-n // t) for n, t in zip(self.Nx_padded, tile))[::-1]
        shared = self._fused_shared()
        kernel.max_dynamic_shared_size_bytes = shared
        args = [None, None, None, *self.stress_material(mat)]
        args += [self.dtype(self.density), *self.damping_args(mat)]
        args += [face_mask(self, Clamped), np.int32(self.comp_stride)]
        args += [self.dtype(1.0 / d) for d in self.dx]
        args += axis_geometry(self, self.step_factors())

        def fd_step(u0, u1, u2):
            args[0], args[1], args[2] = u0, u1, u2
            kernel(grid, tile[::-1], args, shared_mem=shared)
            return u2

        fd_step.memory = []
        return fd_step

    def _fused_shared(self) -> int:
        """Bytes of the fused step's stresses: its tile and a ring a radius wide."""
        ring = math.prod(t + self.space_order for t in FUSED_TILE[self.ndim - 2])
        return self.nvoigt * ring * self.dtype().itemsize

    def _pml_fold(self, adjoint: bool) -> tuple[list, list]:
        """The step kernels' pml arguments and the memory per lined axis."""
        slabs = {d: rest for d, *rest in self.pml_slabs()}
        unlined = cp.zeros(1, dtype=self.dtype)
        fold, memory = [], []
        for d in range(self.ndim):
            if d not in slabs:
                fold += [unlined, unlined, np.int32(0), np.int32(0)]
                continue
            lo, hi, profile, plane = slabs[d]
            # the strain then the divergence memory per component, the adjoint's
            # strain memory twice: read off one block and settled into the other
            rows = (3 if adjoint else 2) * self.ndim
            aux = cp.zeros((rows, (lo + hi) * plane), dtype=self.dtype)
            memory.append(aux)
            fold += [aux, profile, np.int32(lo), np.int32(hi)]
        return fold + [np.int32(1 if adjoint else 0)], memory

    def step_args(self, mat: dict) -> tuple[cpt.NDArray, list, list]:
        """A stress scratch and the argument lists of the stress and update kernels on it."""
        # the stress scratch outlives the closure, its ghost region never written
        sigma = cp.zeros((self.nvoigt, *self.Nx_padded), dtype=self.dtype)
        clamped = face_mask(self, Clamped)
        inv_dx = [self.dtype(1.0 / d) for d in self.dx]
        sargs = [None, sigma, *self.stress_material(mat), clamped]
        sargs += [np.int32(self.comp_stride), *axis_geometry(self, inv_dx)]
        uargs = [None, None, None, sigma, mat["gamma"], self.dtype(self.density)]
        uargs += [*self.damping_args(mat), clamped, np.int32(self.comp_stride)]
        uargs += axis_geometry(self, self.step_factors())
        return sigma, sargs, uargs

    def damping_args(self, mat: dict) -> list:
        """The damping field and the step it is scaled by, if there is one."""
        if self.damping is None:
            return []
        return [mat["damping"], self.dtype(self.dt)]

    def stress_material(self, mat: dict) -> list:
        """The design field and the two Lame parameters, as the kernels take them."""
        return [mat["gamma"], self.dtype(self.lame), self.dtype(self.shear)]

    def excitation_weights(
        self, mat: dict, lin_index: cpt.NDArray[cp.int32]
    ) -> cpt.NDArray:
        """Source weights `dt**2 / (inertia V)`: the kernel inertia leaves the volume out."""
        volume = self.dtype(1.0 / float(np.prod(self.dx)))
        weight = self.point_inverse_inertia(mat["gamma"], lin_index) * volume
        if self.damping is not None:
            node = lin_index % np.int32(self.comp_stride)
            beta = 0.5 * weight * mat["damping"].ravel()[node] * self.dt
            weight = weight / (1.0 + beta)
        return (self.dtype(self.dt**2 * self.source_factor()) * weight).astype(
            self.dtype
        )

    def parametrization_jacobian(self, indicator: cpt.NDArray) -> tuple:
        """Gamma scales inertia and stiffness alike, so both derivatives are 1."""
        return 1.0, 1.0

    def step_factors(self) -> list:
        """Per-axis update factors `dt**2 / dx`, the strain carrying the other `1 / dx`."""
        return [self.dtype(self.dt**2 / d) for d in self.dx]

    def source_factor(self) -> float:
        """Source scaling, unscaled since rho0 is already folded into the point inertia."""
        return 1.0

    def adjoint_weights(self, sensors: cpt.NDArray[cp.int32]) -> cpt.NDArray:
        """`1 / V`: dJ/du is nodal, so it undoes the volume the source weights divide by."""
        volume = self.dtype(1.0 / float(np.prod(self.dx)))
        return cp.full(sensors.shape[1], volume, dtype=self.dtype)

    def gradient_fields(self, mat: dict) -> dict[str, cpt.NDArray]:
        """Accumulators on the component and stress points, plus the design they chain to."""
        grads = {
            "mass": cp.zeros((self.ncomp, *self.Nx_padded), dtype=self.dtype),
            "normal": cp.zeros(self.Nx_padded, dtype=self.dtype),
        }
        if self.ndim > 1:
            grads["shear"] = cp.zeros((self.npairs, *self.Nx_padded), dtype=self.dtype)
        # the point averages have a design dependent chain rule, so keep the field
        grads["design"] = mat["gamma"]
        return grads

    def finalize_gradients(self, grads: dict, kernels: cp.RawModule) -> dict:
        """Chain the point densities through the averages onto the nodal design field."""
        gamma = grads["design"]
        g_mass = cp.zeros(self.Nx_padded, dtype=self.dtype)
        for c in range(self.ncomp):
            density = grads["mass"][c] * wall_weights(self, (c,)) * self.density
            g_mass += point_average_adjoint(self, density, c)
        # the normal density sits on the nodes, so its chain rule is the weight alone
        g_stiff = apply_cell_weights(self, grads["normal"].copy())
        for p, axes in enumerate(PAIRS[self.ndim][self.ndim :]):
            density = grads["shear"][p] * wall_weights(self, axes)
            g_stiff += pair_average_adjoint(self, density, gamma, axes)
        return {"mass": g_mass, "stiff": g_stiff}

    def _density_args(self, grads: dict) -> list:
        """The accumulator and material head shared by the gradient and Frechet closures."""
        args = [grads["mass"], grads["normal"]]
        if self.ndim > 1:
            args.append(grads["shear"])
        return args

    def define_gradient(
        self, kernels: cp.RawModule, mat: dict, grads: dict
    ) -> Callable:
        """Closure accumulating the point mass and stress-point stiffness densities."""
        gradient_kernel = kernels.get_function("gradient_kernel")
        grid, block = grid_block(self)
        inv_dx = [self.dtype(1.0 / d) for d in self.dx]
        args = (
            self._density_args(grads)
            + [None, None, None, None]
            + [
                self.dtype(self.lame),
                self.dtype(self.shear),
                self.dtype(1.0 / self.dt**2),
                face_mask(self, Clamped),
                np.int32(self.comp_stride),
                *axis_geometry(self, inv_dx),
            ]
        )
        head = len(self._density_args(grads))

        def gradient_step(u0, u1, u2, l1):
            args[head], args[head + 1] = u0, u1
            args[head + 2], args[head + 3] = u2, l1
            gradient_kernel(grid, block, args)

        return gradient_step

    def define_frechet(
        self, kernels: cp.RawModule, accs: dict, sign: float
    ) -> Callable:
        """Closure accumulating both quadratic densities of one field triplet, times `sign`."""
        frechet_kernel = kernels.get_function("frechet_kernel")
        grid, block = grid_block(self)
        inv_dx = [self.dtype(1.0 / d) for d in self.dx]
        args = (
            self._density_args(accs)
            + [None, None, None]
            + [
                self.dtype(self.lame),
                self.dtype(self.shear),
                self.dtype(sign / (2.0 * self.dt) ** 2),
                self.dtype(-sign),
                face_mask(self, Clamped),
                np.int32(self.comp_stride),
                *axis_geometry(self, inv_dx),
            ]
        )
        head = len(self._density_args(accs))

        def frechet_step(u0, u1, u2):
            args[head], args[head + 1], args[head + 2] = u0, u1, u2
            frechet_kernel(grid, block, args)

        return frechet_step
