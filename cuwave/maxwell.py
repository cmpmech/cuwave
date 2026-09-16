"""Maxwell's equations in second-order curl-curl form, and their scalar 2D reductions.

`MaxwellWave` is the Yee lattice written as `-C^T nu C` with a diagonal permittivity, which
is the staggered `ElasticWave` layout with its normal-stress block deleted, and it is what 3D
needs. Out of plane one component survives and the curl-curl collapses to a flux
divergence, so 2D needs neither: `ElectricWave` carries E_z with the permittivity as its
inertia and `MagneticWave` carries H_z with the inverse permittivity as its stiffness, both
`PressureWave` on the scalar kernels.
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

from .boundary import Conductor, Magnetic, faces_with
from .scalar import PressureWave
from .wave import (
    PAIRS,
    Simulation,
    apply_cell_weights,
    axis_geometry,
    component_weights,
    grid_block,
    pair_average,
    pair_average_adjoint,
    pair_weights,
    point_average,
    point_average_adjoint,
)

KERNEL_PATH = Path(__file__).parent / "kernels" / "maxwell.cu"
SENSITIVITY_PATH = Path(__file__).parent / "kernels" / "maxwell_sensitivity.cu"


# -------------------------------------- helpers --------------------------------------
def interpolate(indicator: cpt.NDArray, first: float, second: float) -> cpt.NDArray:
    """Two-phase linear interpolation `first + gamma (second - first)` of a material."""
    return first + indicator * (second - first)


# -------------------------------- discretization setup -------------------------------
@dataclass
class PolarizedWave(PressureWave):
    """Two-phase 2D Maxwell base: one out-of-plane component, gamma interpolating epsilon."""

    permittivity1: float = None  # background, gamma = 0
    permittivity2: float = None  # design material, gamma = 1
    permeability: float = 1.0  # the media are non-magnetic, so it is not designed

    def __post_init__(self) -> None:
        """Validate the two phases and the out-of-plane reduction, on top of `Simulation`."""
        super().__post_init__()
        if None in (self.permittivity1, self.permittivity2):
            raise ValueError("PolarizedWave requires permittivity1 and permittivity2")
        if min(self.permittivity1, self.permittivity2, self.permeability) <= 0.0:
            raise ValueError(
                f"permittivity and permeability must be positive: "
                f"{self.permittivity1}, {self.permittivity2}, {self.permeability}"
            )
        if self.ndim > 2:
            raise ValueError(f"an out-of-plane reduction is 1D or 2D, not {self.ndim}D")

    @property
    def light_speed(self) -> float:
        """Fastest speed on the grid, `1 / sqrt(min(eps) mu)`: what bounds the timestep."""
        # the background is the fast phase here, the opposite of the seismic case
        return 1.0 / math.sqrt(
            min(self.permittivity1, self.permittivity2) * self.permeability
        )

    def permittivity(self, indicator: cpt.NDArray) -> cpt.NDArray:
        """Two-phase permittivity `eps1 + gamma (eps2 - eps1)`, linear in the design."""
        return interpolate(indicator, self.permittivity1, self.permittivity2)

    def constant(self, value: float) -> cpt.NDArray:
        """`value` over the padded grid, for the coefficient the design does not enter."""
        return cp.full(self.Nx_padded, value, dtype=self.dtype)

    def step_factors(self) -> list:
        """Per-axis finite-difference step factors `2 * dt**2 / dx**2`."""
        return [self.dtype(2.0 * self.dt**2 / dxk**2) for dxk in self.dx]

    def source_factor(self) -> float:
        """Source scaling, unscaled since both materials already sit in the fields."""
        return 1.0


@dataclass
class ElectricWave(PolarizedWave):
    """Out-of-plane electric field, the permittivity its inertia and `1 / mu` its stiffness.

    The polarization Christiansen & Sigmund 2021 label TE, and the one a waveguide
    device is designed in, since a guided mode's electric field is the one held
    continuous across the sidewalls.
    """

    def parametrization(
        self, indicator: cpt.NDArray
    ) -> tuple[cpt.NDArray, cpt.NDArray]:
        """`indicator` interpolates the permittivity, which is the inertia of E_z."""
        nu = self.constant(1.0 / self.permeability)
        return nu, 1.0 / self.permittivity(indicator)

    def parametrization_jacobian(self, indicator: cpt.NDArray) -> tuple:
        """The inertia is affine in gamma and the stiffness does not depend on it at all."""
        return self.permittivity2 - self.permittivity1, 0.0


@dataclass
class MagneticWave(PolarizedWave):
    """Out-of-plane magnetic field, `mu` its inertia and the inverse permittivity its stiffness.

    The polarization Christiansen & Sigmund 2021 label TM, and the one their metalens is
    designed in. The design enters the stiffness as `1 / eps`, so its jacobian is a field
    rather than the constant `ElectricWave` gets.
    """

    def parametrization(
        self, indicator: cpt.NDArray
    ) -> tuple[cpt.NDArray, cpt.NDArray]:
        """`indicator` interpolates the permittivity, which enters H_z as `1 / eps`."""
        nu = self.constant(1.0 / self.permeability)
        return 1.0 / self.permittivity(indicator), nu

    def parametrization_jacobian(self, indicator: cpt.NDArray) -> tuple:
        """The inertia is constant; `1 / eps` is not affine in gamma, so its derivative is nodal."""
        contrast = self.permittivity2 - self.permittivity1
        return 0.0, -contrast / self.permittivity(indicator) ** 2


# ---------------------------------- the vector case ----------------------------------
@dataclass
class MaxwellWave(Simulation):
    """Curl-curl Maxwell on the Yee lattice, the staggered scheme 3D needs.

    Component `c` of the electric field lives half a node up axis `c` and the curl of the
    pair `(k, l)` half a node up both of its axes, which is the `ElasticWave` displacement
    and shear-stress layout exactly. The operator is assembled as the
    variational derivative of the magnetic energy, so it is `-C^T nu C` with the
    Levi-Civita signs squared away, symmetric by construction and the exact transpose at
    every order.
    """

    permeability: float = 1.0  # background mu, designed only where `magnetic` is set

    kernel_path = KERNEL_PATH
    sensitivity_path = SENSITIVITY_PATH
    default_boundary = Conductor
    gradient_names = ("mass", "stiff")

    magnetic = False  # set where the permeability carries the design too

    @property
    def compile_flags(self) -> tuple[str, ...]:
        """`-DUSE_MAGNETIC` on top of the damping flag, where `mu` carries the design too."""
        flags = super().compile_flags
        return (*flags, "-DUSE_MAGNETIC") if self.magnetic else flags

    @property
    def ncomp(self) -> int:
        """One electric field component per axis."""
        return self.ndim

    @property
    def component_offsets(self) -> npt.NDArray[np.float64]:
        """Component `c` sits half a node up axis `c`: the Yee staggering itself."""
        return 0.5 * np.eye(self.ndim)

    @property
    def npairs(self) -> int:
        """Curl components, one per axis pair: 1 in 2D and 3 in 3D."""
        return len(PAIRS[self.ndim]) - self.ndim

    @property
    def radius(self) -> int:
        """Half the stencil width, so `space_order` is twice it."""
        return self.space_order // 2

    @property
    def reach(self) -> int:
        """The curl reach compounds with the transposed one: `2 radius - 1` nodes."""
        return 2 * self.radius - 1

    def __post_init__(self) -> None:
        """Validate the dimension and the faces, on top of `Simulation.__post_init__`."""
        super().__post_init__()
        if self.ndim < 2:
            raise ValueError(
                "a single in-plane component has no curl; use ElectricWave"
            )
        if self.permeability <= 0.0:
            raise ValueError(f"permeability must be positive: {self.permeability}")
        for pair in self.boundary:
            for condition in pair:
                if condition not in (Conductor, Magnetic):
                    raise ValueError(f"maxwell faces are Conductor or Magnetic: {pair}")

    def inverse_inertia(self, indicator: cpt.NDArray) -> cpt.NDArray:
        """Nodal `1 / (eps W)`, what a sponge scales its conductivity by."""
        permittivity, _ = self.parametrization(indicator)
        mass = permittivity * apply_cell_weights(
            self, cp.ones(self.Nx_padded, dtype=self.dtype)
        )
        return 1.0 / cp.maximum(mass, cp.finfo(self.dtype).tiny)

    def build_materials(self, indicator: cpt.NDArray) -> dict:
        """Point inverse permittivity per component, and the pair inverse permeability."""
        permittivity, nu = self.parametrization(indicator)
        permittivity = cp.ascontiguousarray(permittivity, dtype=self.dtype)
        minv = cp.zeros((self.ncomp, *self.Nx_padded), dtype=self.dtype)
        for c in range(self.ncomp):
            mass = component_weights(self, c) * point_average(self, permittivity, c)
            minv[c] = 1.0 / cp.maximum(mass, cp.finfo(self.dtype).tiny)
        # a conductor holds the tangential field, which is what a zeroed inertia does
        for face in faces_with(self, Conductor):
            wall = [slice(None)] * self.ndim
            wall[face // 2] = 1 if face % 2 == 0 else self.Nx[face // 2] - 2
            for c in range(self.ncomp):
                if c != face // 2:
                    minv[(c, *wall)] = 0.0
        mat = {"minv": cp.ascontiguousarray(minv), "eps": permittivity}
        if self.magnetic:
            nu = cp.ascontiguousarray(nu, dtype=self.dtype)
            pairs = cp.zeros((self.npairs, *self.Nx_padded), dtype=self.dtype)
            for p, axes in enumerate(PAIRS[self.ndim][self.ndim :]):
                pairs[p] = pair_weights(self, axes) * pair_average(self, nu, axes)
            mat["nu_pair"] = cp.ascontiguousarray(pairs)
            mat["nu"] = nu
        if self.damping is not None:
            mat["damping"] = self.damping
        return mat

    def define_step(self, kernels: cp.RawModule, mat: dict) -> Callable:
        """Closure launching the curl kernel and then the update over (u0, u1, u2)."""
        curl_kernel = kernels.get_function("curl_kernel")
        fd_kernel = kernels.get_function("fd_kernel")
        grid, block = grid_block(self)
        # the curl scratch outlives the closure, its invalid pair points never written
        h = cp.zeros((self.npairs, *self.Nx_padded), dtype=self.dtype)
        inv_dx = [self.dtype(1.0 / d) for d in self.dx]
        cargs = [None, h]
        if self.magnetic:
            cargs.append(mat["nu_pair"])
        cargs += [
            self.dtype(1.0 / self.permeability),
            np.int32(self.comp_stride),
            *axis_geometry(self, inv_dx),
        ]
        uargs = [None, None, None, h, mat["minv"]]
        if self.damping is not None:
            uargs += [mat["damping"], self.dtype(self.dt)]
        uargs += [np.int32(self.comp_stride), *axis_geometry(self, self.step_factors())]

        def fd_step(u0, u1, u2):
            cargs[0] = u1
            curl_kernel(grid, block, cargs)
            uargs[0], uargs[1], uargs[2] = u0, u1, u2
            fd_kernel(grid, block, uargs)
            return u2

        return fd_step

    def excitation_weights(
        self, mat: dict, lin_index: cpt.NDArray[cp.int32]
    ) -> cpt.NDArray:
        """Source weights `dt**2 / (eps V)`: the kernel inertia leaves the volume out."""
        volume = self.dtype(1.0 / float(np.prod(self.dx)))
        weight = mat["minv"].ravel()[lin_index] * volume
        if self.damping is not None:
            node = lin_index % np.int32(self.comp_stride)
            beta = 0.5 * weight * mat["damping"].ravel()[node] * self.dt
            weight = weight / (1.0 + beta)
        return (self.dtype(self.dt**2 * self.source_factor()) * weight).astype(
            self.dtype
        )

    def adjoint_weights(self, sensors: cpt.NDArray[cp.int32]) -> cpt.NDArray:
        """`1 / V`: dJ/du is nodal, so it undoes the volume the source weights divide by."""
        volume = self.dtype(1.0 / float(np.prod(self.dx)))
        return cp.full(sensors.shape[1], volume, dtype=self.dtype)

    def step_factors(self) -> list:
        """Per-axis update factors `dt**2 / dx`, the curl carrying the other `1 / dx`."""
        return [self.dtype(self.dt**2 / d) for d in self.dx]

    def source_factor(self) -> float:
        """Source scaling, unscaled since the permittivity is already in the point inertia."""
        return 1.0

    def gradient_fields(self, mat: dict) -> dict[str, cpt.NDArray]:
        """Accumulators on the component points, and on the pair points where `mu` is designed."""
        grads = {
            "mass": cp.zeros((self.ncomp, *self.Nx_padded), dtype=self.dtype),
        }
        if self.magnetic:
            grads["nu"] = cp.zeros((self.npairs, *self.Nx_padded), dtype=self.dtype)
            # the harmonic mean has a design dependent chain rule, so keep the field
            grads["design"] = mat["nu"]
        return grads

    def finalize_gradients(self, grads: dict, kernels: cp.RawModule) -> dict:
        """Chain the point densities through the averages onto the nodal design fields."""
        g_mass = cp.zeros(self.Nx_padded, dtype=self.dtype)
        for c in range(self.ncomp):
            density = grads["mass"][c] * component_weights(self, c)
            g_mass += point_average_adjoint(self, density, c)
        # a non-magnetic medium has no stiffness design dependence, and says so with zeros
        g_stiff = cp.zeros(self.Nx_padded, dtype=self.dtype)
        if self.magnetic:
            for p, axes in enumerate(PAIRS[self.ndim][self.ndim :]):
                density = grads["nu"][p] * pair_weights(self, axes)
                g_stiff += pair_average_adjoint(self, density, grads["design"], axes)
        return {"mass": g_mass, "stiff": g_stiff}

    def _density_args(self, grads: dict) -> list:
        """The accumulator head shared by the gradient and Frechet closures."""
        args = [grads["mass"]]
        if self.magnetic:
            args.append(grads["nu"])
        return args

    def define_gradient(
        self, kernels: cp.RawModule, mat: dict, grads: dict
    ) -> Callable:
        """Closure accumulating the point permittivity density from a triplet and `l1`."""
        gradient_kernel = kernels.get_function("gradient_kernel")
        grid, block = grid_block(self)
        inv_dx = [self.dtype(1.0 / d) for d in self.dx]
        head = self._density_args(grads)
        args = (
            head
            + [None, None, None, None]
            + [
                self.dtype(1.0 / self.dt**2),
                np.int32(self.comp_stride),
                *axis_geometry(self, inv_dx),
            ]
        )
        start = len(head)

        def gradient_step(u0, u1, u2, l1):
            args[start], args[start + 1] = u0, u1
            args[start + 2], args[start + 3] = u2, l1
            gradient_kernel(grid, block, args)

        return gradient_step

    def define_frechet(
        self, kernels: cp.RawModule, accs: dict, sign: float
    ) -> Callable:
        """Closure accumulating the quadratic densities of one field triplet, times `sign`."""
        frechet_kernel = kernels.get_function("frechet_kernel")
        grid, block = grid_block(self)
        inv_dx = [self.dtype(1.0 / d) for d in self.dx]
        head = self._density_args(accs)
        args = (
            head
            + [None, None, None]
            + [
                self.dtype(sign / (2.0 * self.dt) ** 2),
                self.dtype(-sign),
                np.int32(self.comp_stride),
                *axis_geometry(self, inv_dx),
            ]
        )
        start = len(head)

        def frechet_step(u0, u1, u2):
            args[start], args[start + 1], args[start + 2] = u0, u1, u2
            frechet_kernel(grid, block, args)

        return frechet_step


@dataclass
class DielectricWave(MaxwellWave):
    """Two-phase dielectric, gamma interpolating the permittivity between the two phases."""

    permittivity1: float = None  # background, gamma = 0
    permittivity2: float = None  # design material, gamma = 1

    def __post_init__(self) -> None:
        """Validate the two phases, on top of `MaxwellWave.__post_init__`."""
        super().__post_init__()
        if None in (self.permittivity1, self.permittivity2):
            raise ValueError("DielectricWave requires permittivity1 and permittivity2")
        if min(self.permittivity1, self.permittivity2) <= 0.0:
            raise ValueError(
                f"permittivity must be positive: "
                f"{self.permittivity1}, {self.permittivity2}"
            )

    @property
    def light_speed(self) -> float:
        """Fastest speed on the grid, `1 / sqrt(min(eps) mu)`: what bounds the timestep."""
        return 1.0 / math.sqrt(
            min(self.permittivity1, self.permittivity2) * self.permeability
        )

    def parametrization(
        self, indicator: cpt.NDArray
    ) -> tuple[cpt.NDArray, cpt.NDArray | None]:
        """`indicator` interpolates the permittivity; the permeability is left uniform."""
        return interpolate(indicator, self.permittivity1, self.permittivity2), None

    def parametrization_jacobian(self, indicator: cpt.NDArray) -> tuple:
        """The permittivity is affine in gamma, and the permeability does not depend on it."""
        return self.permittivity2 - self.permittivity1, 0.0
