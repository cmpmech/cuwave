"""Boundary conditions, one per (axis, side).

A `BoundaryCondition` is a declarative marker: it names a kernel and nothing more, so a
setup can place one per face in `Simulation.boundary` long before the module is
compiled. `define_boundary` turns the markers into the launch closures once
`compile_kernels` has run, which is why `simulate` needs no separate preparation phase.

`sponge` and `Pml` open the domain from the other side: they absorb in the material
behind a face rather than acting on the ghost ring. The sponge is a damping field; the
PML is declarative like a condition, its profiles built once the step is defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import cupy.typing as cpt

    from .wave import Simulation


@dataclass(frozen=True, repr=False)
class BoundaryCondition:
    """Names a kernel in the equation's .cu, or nothing. Frozen, so it can key the grouping."""

    kernel: str | None
    name: str = ""

    def __repr__(self) -> str:
        # so that printing a Simulation shows the condition, not the kernel name
        return self.name or self.kernel.removesuffix("_kernel")

    def define(
        self, sim: Simulation, kernels: cp.RawModule, faces: Sequence[int]
    ) -> Callable[[cpt.NDArray], None]:
        """closure applying this condition on `faces`, the codes 2 * axis + side"""
        kernel = kernels.get_function(self.kernel)
        threads = 256

        # one thread per boundary node, edges and corners excluded
        inner = [n - 2 for n in sim.Nx]
        num_boundary = sum(
            int(np.prod(inner[:d] + inner[d + 1 :])) for d in (f // 2 for f in faces)
        )
        grid = ((num_boundary + threads - 1) // threads,)

        # a bitmask, so the kernel keeps its axis loop at compile time
        mask = np.int32(sum(1 << f for f in faces))

        geom = [sim.Nx[0]]
        for d in range(1, sim.ndim):
            geom += [sim.Nx[d], sim.strides[d - 1]]

        args = [None, mask, *geom]

        def step(u):
            args[0] = u
            kernel(grid, (threads,), args)

        return step


Neumann = BoundaryCondition("homogeneous_neumann_kernel")
Dirichlet = BoundaryCondition("homogeneous_dirichlet_kernel")
# an elastic wall is the interior-cell assembly or a zeroed inertia: no kernel needed
Traction = BoundaryCondition(None, "traction")
Clamped = BoundaryCondition(None, "clamped")
# the same walls read electromagnetically, the tangential field held or left natural
Conductor = BoundaryCondition(None, "conductor")
Magnetic = BoundaryCondition(None, "magnetic")


def faces_with(sim: Simulation, condition: BoundaryCondition) -> list[int]:
    """The `2 * axis + side` codes of the faces `sim.boundary` marks with `condition`."""
    return [
        2 * d + side
        for d, pair in enumerate(sim.boundary)
        for side, marker in enumerate(pair)
        if marker == condition  # a copied or unpickled condition is still the same
    ]


def face_mask(sim: Simulation, condition: BoundaryCondition) -> np.int32:
    """The faces carrying `condition`, as the bitmask the kernels take."""
    return np.int32(sum(1 << f for f in faces_with(sim, condition)))


def canonical_boundary(
    boundary: BoundaryCondition | Sequence | None,
    ndim: int,
    default: BoundaryCondition = Neumann,
) -> tuple[tuple[BoundaryCondition, BoundaryCondition], ...]:
    """makes the boundary canonical ((low, high),) * ndim

    Accepts `None` for the equation's `default` on every face.
    """
    if boundary is None:
        boundary = default
    if isinstance(boundary, BoundaryCondition):
        boundary = (boundary,) * ndim
    if len(boundary) != ndim:
        raise ValueError(f"boundary needs one (low, high) pair per axis: {ndim}")
    pairs = []
    for pair in boundary:
        if isinstance(pair, BoundaryCondition):
            pair = (pair, pair)
        if len(pair) != 2:
            raise ValueError("a boundary axis is a (low, high) pair of conditions")
        pairs.append(tuple(pair))
    return tuple(pairs)


def define_boundary(
    sim: Simulation, kernels: cp.RawModule
) -> Callable[[cpt.NDArray], cpt.NDArray]:
    """one launch per distinct condition, so the default stays one launch."""
    groups = {}
    for d, pair in enumerate(sim.boundary):
        for side, condition in enumerate(pair):
            if condition.kernel is not None:
                groups.setdefault(condition, []).append(2 * d + side)
    steps = [
        condition.define(sim, kernels, faces) for condition, faces in groups.items()
    ]

    def bc_step(u):
        for step in steps:
            step(u)
        return u

    return bc_step


def _canonical_faces(ndim: int, faces: Sequence[int] | None) -> tuple[int, ...]:
    """Expand `None` to every face and check the codes fit `ndim`."""
    faces = tuple(range(2 * ndim) if faces is None else faces)
    if any(f not in range(2 * ndim) for f in faces):
        raise ValueError(f"face codes run to {2 * ndim - 1} in {ndim}D: {faces}")
    return faces


def _validate_layer(sim: Simulation, width: int, faces: Sequence[int] | None) -> tuple:
    """Check a layer fits behind `faces` and expand `None` to every face."""
    if width < 1:
        raise ValueError(f"width must be at least one node: {width}")
    faces = _canonical_faces(sim.ndim, faces)
    for d in range(sim.ndim):
        sides = sum(2 * d + side in faces for side in (0, 1))
        if sides * width >= sim.Nx[d] - 2:
            raise ValueError(
                f"{sides} layer(s) of {width} nodes leave no interior on axis {d}: "
                f"Nx={sim.Nx[d]}"
            )
    return faces


def _layer_taper(sim: Simulation, width: int, faces: Sequence[int]) -> cpt.NDArray:
    """Linear ramp over the `width` nodes behind `faces`, 1 on the wall and 0 inside."""
    taper = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    interior = cp.ones(sim.Nx_padded, dtype=bool)
    for d in range(sim.ndim):
        shape = [1] * sim.ndim
        shape[d] = sim.Nx_padded[d]
        index = cp.arange(sim.Nx_padded[d], dtype=sim.dtype).reshape(shape)
        interior &= (index >= 1) & (index <= sim.Nx[d] - 2)
        for side in (0, 1):
            if 2 * d + side in faces:
                wall = 1 if side == 0 else sim.Nx[d] - 2
                ramp = cp.clip(1.0 - cp.abs(index - wall) / width, 0.0, 1.0)
                # max, so a corner is graded once rather than by each of its faces
                taper = cp.maximum(taper, ramp)
    # the ghost ring is slaved to its mirror and the padding tail is never read
    return taper * interior


def pad_for_sponge(
    Nx: tuple[int, ...],
    dx: tuple[float, ...],
    thickness: float,
    faces: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], int, tuple[float, ...], tuple[slice, ...]]:
    """Grow a region of interest `Nx` by a sponge layer of physical `thickness`.

    The layer is grid the simulation carries but the application does not own, so what
    a driver needs back is where its region of interest ends up: `origin` shifts the
    coordinates it places transducers and defects at, and `region` selects it out of
    the grown grid for a design mask or a figure.

    Args:
        Nx: logical grid points per axis of the region of interest, ghost nodes
            included.
        dx: grid spacing per axis.
        thickness: layer depth in physical units, taken in nodes off the finest axis so
            that no face comes out thinner than asked for.
        faces: the `2 * axis + side` codes to line, or None for every face.

    Returns:
        (Nx, width, origin, region): the grown extent to build the simulation on, the
        layer `width` in nodes to hand `sponge`, the physical origin of the region of
        interest per axis, and the index tuple selecting its interior nodes.
    """
    if thickness <= 0.0:
        raise ValueError(f"thickness must be positive: {thickness}")
    faces = _canonical_faces(len(Nx), faces)
    width = round(thickness / min(dx))
    if width < 1:
        raise ValueError(f"thickness {thickness} is under one node at dx {min(dx)}")
    pads = [
        tuple(width if 2 * d + side in faces else 0 for side in (0, 1))
        for d in range(len(Nx))
    ]
    grown = tuple(n + lo + hi for n, (lo, hi) in zip(Nx, pads))
    origin = tuple(lo * h for (lo, _), h in zip(pads, dx))
    region = tuple(slice(lo + 1, n - 1 - hi) for (lo, hi), n in zip(pads, grown))
    return grown, width, origin, region


def sponge(
    sim: Simulation,
    indicator: cpt.NDArray,
    width: int,
    beta: float,
    faces: Sequence[int] | None = None,
) -> cpt.NDArray:
    """Damping field ramping to `beta` over the `width` nodes behind a face, 0 elsewhere.

    The price is losslessness: the field is rejected by `superposition_sensitivity`,
    whose reverse-time reconstruction needs an operator it can run backwards, and it is
    what `reconstruction_sensitivity` records a strip of in order to march past it.

    Args:
        sim: the simulation the field is built for, whose `dt` and inertia scale it.
        indicator: the design field, read for the inertia the damping is scaled by.
        width: layer thickness in nodes, measured inward from the wall node.
        beta: peak `d * dt / 2m` at the wall, the dimensionless decay per step.
        faces: the `2 * axis + side` codes to line, or None for every face.

    Returns:
        the field to set as `Simulation.damping`, over the padded grid, ramped
        quadratically so the layer front is not a coherent reflector.
    """
    if beta < 0.0:
        raise ValueError(f"beta must be non-negative: {beta}")
    faces = _validate_layer(sim, width, faces)
    minv = sim.inverse_inertia(indicator)
    taper = _layer_taper(sim, width, faces)
    return cp.ascontiguousarray(
        (2.0 * beta * taper**2 / (sim.dt * minv)).astype(sim.dtype)
    )


@dataclass(frozen=True)
class Pml:
    """A convolutional perfectly matched layer behind `faces`, for `Simulation.pml`.

    Every derivative along an axis carries a memory variable inside the `width` nodes
    behind that axis' lined faces, its graded profile set by the target `reflection`
    at the fastest `wavespeed` (Komatitsch & Martin 2007,
    https://doi.org/10.1190/1.2757586).
    """

    width: int  # layer depth in nodes, the wall node included
    wavespeed: float  # fastest wave speed in the layer, which sizes the damping
    reflection: float = 1e-6  # target reflection at normal incidence
    frequency: float = 0.0  # CFS shift pi * frequency through the layer, 0 for none
    faces: tuple[int, ...] | None = None  # the `2 * axis + side` codes, None for all


def validate_pml(sim: Simulation) -> None:
    """Raise if `sim.pml` cannot be stepped: unsupported, combined, or too wide."""
    layer = sim.pml
    if not sim.accepts_pml:
        raise NotImplementedError(f"{type(sim).__name__} has no pml; use sponge")
    if sim.damping is not None or sim.masked:
        raise ValueError("a pml takes neither a damping field nor a domain mask")
    least = 2 + sim.pml_margin
    if layer.width < least or layer.wavespeed <= 0.0 or not 0.0 < layer.reflection < 1:
        raise ValueError(
            f"a pml needs width >= {least}, a speed and 0 < reflection < 1: {layer}"
        )
    if sim.pml_shifted and layer.frequency <= 0.0:
        raise ValueError(
            f"{type(sim).__name__} keeps a static mode in an unshifted pml: set "
            f"frequency, near the low end of the band"
        )
    # the transposed taps reach `reach` nodes past the layer, which must not meet
    faces = _validate_layer(sim, layer.width + sim.reach, layer.faces)
    if sim.pml_walls is not None:
        for face in faces:
            wall = sim.boundary[face // 2][face % 2]
            if wall not in sim.pml_walls:
                raise ValueError(
                    f"face {face} ends the pml on {wall}, which traps a growing wave: "
                    f"make it {' or '.join(map(repr, sim.pml_walls))}"
                )


def pml_widths(sim: Simulation) -> list[tuple[int, int]]:
    """Per axis, the (low, high) layer depths in nodes, 0 for a face left unlined."""
    faces = _canonical_faces(sim.ndim, sim.pml.faces)
    return [
        tuple(sim.pml.width if 2 * d + side in faces else 0 for side in (0, 1))
        for d in range(sim.ndim)
    ]


def pml_profiles(sim: Simulation) -> list[cpt.NDArray | None]:
    """Per lined axis, the (4, Nx[d]) recursion coefficients `a, b` at nodes and half points.

    Row 0 and 1 are the node values, rows 2 and 3 those of the half point `a + 1/2`,
    graded from 0 at the front to the wall and mirrored across it, so the ghost cell
    sees what its mirror does. An axis with no lined face gets None.
    """
    layer = sim.pml
    # the front node is undamped, and so are the margin nodes in front of it
    L = layer.width - 1 - sim.pml_margin
    profiles = []
    for d, (lo, hi) in enumerate(pml_widths(sim)):
        if not lo + hi:
            profiles.append(None)
            continue
        n = sim.Nx[d]
        damping = (
            -3.0 * layer.wavespeed * np.log(layer.reflection) / (2.0 * L * sim.dx[d])
        )
        rows = []
        for position in (np.arange(n) + shift for shift in (0.0, 0.5)):
            distance = np.full(n, np.inf)
            if lo:
                distance = np.minimum(distance, np.abs(position - 1))
            if hi:
                distance = np.minimum(distance, np.abs(n - 2 - position))
            depth = np.clip((L - distance) / L, 0.0, 1.0)
            d0 = damping * depth**2
            alpha = np.pi * layer.frequency
            inside = depth > 0.0
            b = np.where(inside, np.exp(-(d0 + alpha) * sim.dt), 1.0)
            a = np.where(inside, d0 / np.maximum(d0 + alpha, 1e-300) * (b - 1.0), 0.0)
            rows += [a, b]
        profiles.append(cp.asarray(np.stack(rows), dtype=sim.dtype))
    return profiles


def pml_nodes(sim: Simulation) -> cpt.NDArray:
    """Interior nodes inside the layer, where the step carries memory terms."""
    inside = cp.zeros(sim.Nx_padded, dtype=bool)
    if sim.pml is None:
        return inside
    for d, (lo, hi) in enumerate(pml_widths(sim)):
        index = [slice(None)] * sim.ndim
        for rows in (slice(1, 1 + lo), slice(sim.Nx[d] - 1 - hi, sim.Nx[d] - 1)):
            index[d] = rows
            inside[tuple(index)] = True
    interior = cp.zeros_like(inside)
    interior[tuple(slice(1, n - 1) for n in sim.Nx)] = True
    return inside & interior
