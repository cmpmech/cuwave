"""Scalar pressure wave equations, the nodal sibling of the vector elastic schemes.

One unknown per node and a per-axis flux, so the operator is the cheapest of the three
families and needs no component offsets. `PressureWave` holds the nodal `stiff`/`minv`
fields and the adjoint hooks; `ScalarWave` scales both with one indicator gamma, which
is why `minv` is never formed (`derive_inertia`), and `AcousticWave` interpolates
inverse density and inverse bulk modulus between two phases for topology optimization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
import numpy as np

from .boundary import Neumann
from .wave import (
    Simulation,
    apply_cell_weights,
    grid_block,
    mirror_ghosts,
    sensor_cell_weights,
)

KERNEL_PATH = Path(__file__).parent / "kernels" / "scalar.cu"
SENSITIVITY_PATH = Path(__file__).parent / "kernels" / "scalar_sensitivity.cu"


# -------------------------------- discretization setup -------------------------------
@dataclass
class PressureWave(Simulation):
    """Scalar wave equation base: nodal `stiff`/`minv` material fields, optional damping."""

    kernel_path = KERNEL_PATH
    sensitivity_path = SENSITIVITY_PATH
    default_boundary = Neumann

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

    gradient_names = ("mass", "stiff")  # the fields the adjoint differentiates

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
        # sign is fixed per pass, so it is folded into the factors, not recomputed
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
