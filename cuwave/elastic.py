"""Isotropic elastic wave equation, a sibling of `PressureWave` on the same time march.

The stencil is a 9-point one in 2D and 27-point in 3D, gathered cell by cell rather than
axis by axis, since a variable-coefficient elastic operator couples the components through
mixed derivatives that a per-axis flux cannot carry. Its coefficients are derived as
`-B^T C B` with `C` on the cell, which is what makes the operator exactly symmetric for a
varying material, never differentiates the material, leaves the leapfrog reversible, and
makes a traction-free surface the natural condition of the interior-cell sum.
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

from .boundary import Clamped, Traction
from .wave import Simulation, apply_cell_weights, grid_block

KERNEL_PATH = Path(__file__).parent / "kernels" / "elastic.cu"
SENSITIVITY_PATH = Path(__file__).parent / "kernels" / "elastic_sensitivity.cu"

# Voigt row order per dimension, as (k, l) strain pairs
PAIRS = {
    1: ((0, 0),),
    2: ((0, 0), (1, 1), (0, 1)),
    3: ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)),
}


# -------------------------------------- helpers --------------------------------------
def corner_bits(ndim: int) -> list[tuple[int, ...]]:
    """The `2**ndim` nodes of a cell, node `c` having bit `d` set on the plus side of axis `d`."""
    return [tuple((c >> d) & 1 for d in range(ndim)) for c in range(2**ndim)]


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


def block_offsets(radius: int) -> list[int]:
    """The `2 * radius` node offsets a cell of that radius reaches, low corner at 0."""
    return list(range(-radius + 1, radius + 1))


def block_indices(ndim: int, radius: int) -> list[tuple[int, ...]]:
    """The cell's node block in the kernel's order, axis 0 running fastest.

    Entry `m` of the returned list gives the per-axis position into `block_offsets`, so
    that `m = sum_d position[d] * (2 * radius)**d`. At `radius` 1 this is `corner_bits`.
    """
    width = 2 * radius
    return [
        tuple((m // width**d) % width for d in range(ndim)) for m in range(width**ndim)
    ]


def lagrange_weights(radius: int, t: float) -> tuple[npt.NDArray, npt.NDArray]:
    """Value and index-derivative weights at `t` inside the cell, on `block_offsets`."""
    offs = np.asarray(block_offsets(radius), dtype=float)
    value = np.zeros(len(offs))
    slope = np.zeros(len(offs))
    for i, o in enumerate(offs):
        others = np.delete(offs, i)
        value[i] = np.prod((t - others) / (o - others))
        slope[i] = sum(
            np.prod(
                [
                    (t - others[m]) / (o - others[m])
                    for m in range(len(others))
                    if m != j
                ]
            )
            / (o - others[j])
            for j in range(len(others))
        )
    return value, slope


def cell_stencil(
    ndim: int, dx: tuple[float, ...], C: npt.NDArray, radius: int = 1
) -> npt.NDArray[np.float64]:
    """One cell's contribution to the nodal stencil, ordered (node, component).

    Built as `sum_q w_q B_q^T C B_q` over the cell, `B` the discrete symmetric gradient on
    the `(2 * radius)**ndim` node block, so the assembled operator is symmetric by
    construction rather than by inspection, at any width. At `radius` 1 the entries collapse
    to the 9-point (27-point in 3D) finite difference stencil

        f_x = (lame + 2 mu) [M_y * D2_x] u_x + mu [M_x * D2_y] u_x + (lame + mu) [D_x * D_y] u_y

    with the second difference `D2` = (1, -2, 1), the centred first difference `D` and the
    transverse average `M` = (1, 4, 1) / 6. That average is the only departure from the
    textbook elastic stencil, and it is what removes the checkerboard from the null space:
    one quadrature point per axis would give (1, 2, 1) / 4 and leave it there as an undamped
    hourglass mode. Two points per axis keep it out at every `radius`.

    Args:
        ndim: dimensionality of the cell.
        dx: cell side per axis.
        C: (voigt, voigt) stiffness matrix, from `voigt`.
        radius: half the stencil width, so the order is `2 * radius`.

    Returns:
        the (n * ndim, n * ndim) matrix for `n = (2 * radius)**ndim`, node `c` and
        component `i` occupying row `c * ndim + i`, `c` running over the block in the bit
        order of `corner_bits` generalized to `2 * radius` offsets per axis.
    """
    block = block_indices(ndim, radius)
    nloc = len(block) * ndim
    # exact for the product, and never under two, which is what keeps the checkerboard out
    npoints = max(2, 2 * radius - 1)
    nodes, weights = np.polynomial.legendre.leggauss(npoints)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights * float(np.prod(dx)) ** (1.0 / ndim)
    K = np.zeros((nloc, nloc))
    for point in itertools.product(range(npoints), repeat=ndim):
        weight = float(np.prod([weights[q] for q in point]))
        table = [lagrange_weights(radius, nodes[q]) for q in point]
        dN = np.zeros((len(block), ndim))
        for m, entry in enumerate(block):
            for k in range(ndim):
                term = table[k][1][entry[k]] / dx[k]
                for axis in range(ndim):
                    if axis != k:
                        term *= table[axis][0][entry[axis]]
                dN[m, k] = term
        B = np.zeros((len(PAIRS[ndim]), nloc))
        for r, (axis_k, axis_l) in enumerate(PAIRS[ndim]):
            for m in range(len(block)):
                if axis_k == axis_l:
                    B[r, m * ndim + axis_k] += dN[m, axis_k]
                else:
                    B[r, m * ndim + axis_l] += dN[m, axis_k]  # engineering shear
                    B[r, m * ndim + axis_k] += dN[m, axis_l]
        K += weight * (B.T @ C @ B)
    return K


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
    where `wave.stable_dt` only knows the wave speed and the spacing.

    Args:
        sim: the simulation to measure, whose own `dt` sets the scale of the answer.
        indicator: the design field, which is what a high contrast enters through.
        iterations: power iterations, 60 being ample for three digits.
        safety: fraction of the bound to return.

    Returns:
        the timestep to build `sim` with. A wide stencil over a strong contrast can put
        this far below `wave.stable_dt`, which is the signal to drop `space_order` rather
        than to shrink `dt`.
    """
    from .wave import compile_kernels, define_step_method

    mat = sim.build_materials(indicator)
    step = define_step_method(sim, compile_kernels(sim), mat)
    interior = (Ellipsis, *(slice(1, n - 1) for n in sim.Nx))
    field = cp.asarray(
        np.random.default_rng(0).standard_normal((sim.ncomp, *sim.Nx_padded)),
        dtype=sim.dtype,
    )
    zero = cp.zeros_like(field)
    out = cp.zeros_like(field)
    value = 0.0
    for _ in range(iterations):
        field /= cp.linalg.norm(field)
        out[...] = 0.0
        step(zero, field, out)
        applied = 2.0 * field - out
        masked = cp.zeros_like(applied)
        masked[interior] = applied[interior]
        value = float(cp.sum(field[interior] * masked[interior]))
        field = masked
    if value <= 0.0:
        raise ValueError(f"the operator came back non-positive: {value}")
    return safety * sim.dt * float(np.sqrt(4.0 / value))


def cell_average(sim: Simulation, field: cpt.NDArray) -> cpt.NDArray:
    """Harmonic mean of `field` over the `2**ndim` corners of a cell, at its low corner.

    Harmonic and not arithmetic for the reason the scalar flux uses it: it keeps the cell
    stiffness single valued across a material jump, and it is what holds the stable
    timestep together at a high contrast, where a light node borders a stiff cell.
    """
    out = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    safe = cp.maximum(field, cp.finfo(sim.dtype).tiny)
    inner = tuple(slice(0, n - 1) for n in sim.Nx)
    for corner in corner_bits(sim.ndim):
        shifted = tuple(slice(b, n - 1 + b) for b, n in zip(corner, sim.Nx))
        out[inner] += 1.0 / safe[shifted]
    out[inner] = 2.0**sim.ndim / out[inner]
    return out


# ------------------------------- discretization setup --------------------------------
@dataclass
class ElasticWave(Simulation):
    """Isotropic elasticity, parametrized by a density-scaling indicator gamma.

    Both wave speeds are held fixed and gamma scales the density, so `C = gamma * rho0 * C`
    scales inertia and stiffness alike and the stable timestep does not move with the
    design. This is the parametrization ultrasonic full waveform inversion reaches for,
    since a void is a density contrast at unchanged speeds.
    """

    density: float = None  # background density rho0
    wavespeed_p: float = None  # pressure wave speed
    wavespeed_s: float = None  # shear wave speed
    plane: str = "strain"  # "strain" or "stress", 2D only

    kernel_path = KERNEL_PATH
    sensitivity_path = SENSITIVITY_PATH
    default_boundary = Traction
    gradient_names = ("mass", "stiff", "cell")

    @property
    def ncomp(self) -> int:
        """One displacement component per axis."""
        return self.ndim

    def __post_init__(self) -> None:
        """Validate the material and the 2D plane assumption, then derive the grid."""
        super().__post_init__()
        if None in (self.density, self.wavespeed_p, self.wavespeed_s):
            raise ValueError("ElasticWave requires density, wavespeed_p, wavespeed_s")
        if self.plane not in ("strain", "stress"):
            raise ValueError(f"plane must be strain or stress: {self.plane}")
        if self.ndim == 3 and self.plane != "strain":
            raise ValueError("plane stress is a 2D reduction, not a 3D one")
        self._stencil = None  # built once: the table grows fast with the radius
        if self.wavespeed_s >= self.wavespeed_p:
            raise ValueError(
                f"wavespeed_s must be below wavespeed_p: "
                f"{self.wavespeed_s} >= {self.wavespeed_p}"
            )

    @property
    def radius(self) -> int:
        """Half the stencil width, so `space_order` is twice it."""
        return self.space_order // 2

    @property
    def compile_flags(self) -> tuple[str, ...]:
        """The damping flag, plus the stencil radius the cell gather unrolls over."""
        return (*super().compile_flags, f"-DRADIUS={self.radius}")

    @property
    def lame(self) -> float:
        """First Lame parameter `rho0 * (c_p**2 - 2 c_s**2)`."""
        return self.density * (self.wavespeed_p**2 - 2.0 * self.wavespeed_s**2)

    @property
    def shear(self) -> float:
        """Second Lame parameter `rho0 * c_s**2`."""
        return self.density * self.wavespeed_s**2

    def stencil(self) -> cpt.NDArray:
        """Stencil table at `gamma = 1`, one graded radius per slice, zero padded.

        A cell near a wall carries a lower radius than the interior, so the kernel indexes
        the table by that radius and finds the narrower stencil embedded in the full block,
        its offsets lined up and everything outside its support left at zero.
        """
        if self._stencil is not None:
            return self._stencil
        C = voigt(self.ndim, self.lame, self.shear, self.plane)
        nloc = (2 * self.radius) ** self.ndim * self.ndim
        table = np.zeros((self.radius, nloc, nloc))
        full = block_indices(self.ndim, self.radius)
        for graded in range(1, self.radius + 1):
            K = cell_stencil(self.ndim, self.dx, C, radius=graded)
            rows = [
                full.index(
                    tuple(entry[d] + self.radius - graded for d in range(self.ndim))
                )
                for entry in block_indices(self.ndim, graded)
            ]
            for a, ra in enumerate(rows):
                for b, rb in enumerate(rows):
                    table[graded - 1][
                        ra * self.ndim : (ra + 1) * self.ndim,
                        rb * self.ndim : (rb + 1) * self.ndim,
                    ] = K[
                        a * self.ndim : (a + 1) * self.ndim,
                        b * self.ndim : (b + 1) * self.ndim,
                    ]
        self._stencil = cp.asarray(table.ravel(), dtype=self.dtype)
        return self._stencil

    def cell_weights(self) -> cpt.NDArray:
        """Nodal cell weights W, halved once per wall the node sits on."""
        return apply_cell_weights(self, cp.ones(self.Nx_padded, dtype=self.dtype))

    def clamped_faces(self) -> list[int]:
        """The `2 * axis + side` codes carrying `Clamped`, whose wall nodes are held at zero."""
        return [
            2 * d + side
            for d, pair in enumerate(self.boundary)
            for side, condition in enumerate(pair)
            if condition is Clamped
        ]

    def inverse_inertia(self, indicator: cpt.NDArray) -> cpt.NDArray:
        """Lumped `1 / (gamma rho0 V W)`, the mass the interior-cell assembly implies."""
        volume = float(np.prod(self.dx))
        mass = indicator * (self.density * volume) * self.cell_weights()
        return 1.0 / cp.maximum(mass, cp.finfo(self.dtype).tiny)

    def build_materials(self, indicator: cpt.NDArray) -> dict:
        """Lumped inverse inertia, the cell design field, and the stencil table."""
        minv = self.inverse_inertia(indicator)
        for face in self.clamped_faces():
            wall = [slice(None)] * self.ndim
            wall[face // 2] = 1 if face % 2 == 0 else self.Nx[face // 2] - 2
            minv[tuple(wall)] = 0.0
        mat = {
            "minv": cp.ascontiguousarray(minv, dtype=self.dtype),
            "gamma": cp.ascontiguousarray(indicator, dtype=self.dtype),
            "cell": cell_average(self, indicator),
            "stencil": self.stencil(),
        }
        if self.damping is not None:
            mat["damping"] = self.damping
        return mat

    def step_kernel_args(self, mat: dict) -> tuple:
        """Material, stencil table and component stride for the step kernel."""
        args = (mat["minv"], mat["cell"], mat["stencil"])
        if self.damping is not None:
            args += (mat["damping"], self.dtype(self.dt))
        return args + (np.int32(self.comp_stride),)

    def excitation_weights(
        self, mat: dict, lin_index: cpt.NDArray[cp.int32]
    ) -> cpt.NDArray:
        """Source weights `dt**2 / inertia`, the spatial node read off the folded index."""
        node = lin_index % np.int32(self.comp_stride)
        weight = mat["minv"].ravel()[node]
        if self.damping is not None:
            beta = 0.5 * weight * mat["damping"].ravel()[node] * self.dt
            weight = weight / (1.0 + beta)
        return (self.dtype(self.dt**2 * self.source_factor()) * weight).astype(
            self.dtype
        )

    def parametrization_jacobian(self, indicator: cpt.NDArray) -> tuple:
        """Gamma scales inertia and stiffness alike, so both derivatives are 1."""
        return 1.0, 1.0

    def step_factors(self) -> list:
        """`dt**2`; the grid spacing already sits in the stencil table."""
        return [self.dtype(self.dt**2)] * self.ndim

    def source_factor(self) -> float:
        """Source scaling, unscaled since rho0 is already folded into the lumped inertia."""
        return 1.0

    def axis_geometry(self) -> list:
        """Extents and previous-axis strides, the step kernel's tail without the factors."""
        geom = [self.Nx[0]]
        for d in range(1, self.ndim):
            geom += [self.Nx[d], self.strides[d - 1]]
        return geom

    def gradient_fields(self, mat: dict) -> dict[str, cpt.NDArray]:
        """Nodal accumulators plus the cell one the stiffness density lands in first."""
        grads = {
            name: cp.zeros(self.Nx_padded, dtype=self.dtype)
            for name in self.gradient_names
        }
        # the harmonic cell mean has a design dependent chain rule, so keep both fields
        grads["material"] = mat["cell"]
        grads["design"] = mat["gamma"]
        return grads

    def finalize_gradients(self, grads: dict, kernels: cp.RawModule) -> dict:
        """Weight the mass density by W and spread the cell density onto its corners."""
        apply_cell_weights(self, grads["mass"])
        cell_to_node = kernels.get_function("cell_to_node_kernel")
        grid, block = grid_block(self)
        cell_to_node(
            grid,
            block,
            [
                grads["stiff"],
                grads["cell"],
                grads["material"],
                grads["design"],
                self.dtype(1.0 / 2.0**self.ndim),
                *self.axis_geometry(),
            ],
        )
        return {"mass": grads["mass"], "stiff": grads["stiff"]}

    def define_gradient(
        self, kernels: cp.RawModule, mat: dict, grads: dict
    ) -> Callable:
        """Closure accumulating the nodal mass and cell stiffness densities."""
        gradient_kernel = kernels.get_function("gradient_kernel")
        grid, block = grid_block(self)
        # d(mass)/d(gamma) is rho0 V, the W of the lumping supplied by the epilogue
        mass_factor = self.dtype(self.density * float(np.prod(self.dx)) / self.dt**2)
        args = [grads["mass"], grads["cell"], None, None, None, None] + [
            mat["stencil"],
            mass_factor,
            np.int32(self.comp_stride),
            *self.axis_geometry(),
        ]

        def gradient_step(u0, u1, u2, l1):
            args[2], args[3], args[4], args[5] = u0, u1, u2, l1
            gradient_kernel(grid, block, args)

        return gradient_step

    def define_frechet(
        self, kernels: cp.RawModule, accs: dict, sign: float
    ) -> Callable:
        """Closure accumulating both quadratic densities of one field triplet, times `sign`."""
        frechet_kernel = kernels.get_function("frechet_kernel")
        grid, block = grid_block(self)
        volume = float(np.prod(self.dx))
        args = [accs["mass"], accs["cell"], None, None, None] + [
            self.stencil(),
            self.dtype(sign * self.density * volume / (2.0 * self.dt) ** 2),
            self.dtype(-sign),
            np.int32(self.comp_stride),
            *self.axis_geometry(),
        ]

        def frechet_step(u0, u1, u2):
            args[2], args[3], args[4] = u0, u1, u2
            frechet_kernel(grid, block, args)

        return frechet_step

    def adjoint_weights(self, sensors: cpt.NDArray[cp.int32]) -> cpt.NDArray:
        """Ones: the lumped inertia already carries W, and dJ/du is nodal, not a density."""
        return cp.ones(sensors.shape[1], dtype=self.dtype)
