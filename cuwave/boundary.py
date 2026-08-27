"""Boundary conditions, one per (axis, side).

A `BoundaryCondition` is a declarative marker: it names a kernel and nothing
more, so a setup can place one per face in `Simulation.boundary` long before
the module is compiled. `define_boundary` turns the markers into the launch
closures once `compile_kernels` has run, which is why `simulate` needs no
separate preparation phase.

`random_layer` opens the domain from the other side: it perturbs the material
rather than the ghost ring, so it needs no kernel and no marker of its own.
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
    """Names a kernel in wave.cu. Frozen, so it can key the grouping below."""

    kernel: str

    def __repr__(self) -> str:
        # so that printing a Simulation shows the condition, not the kernel name
        return self.kernel.removesuffix("_kernel")

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


def canonical_boundary(
    boundary: BoundaryCondition | Sequence | None, ndim: int
) -> tuple[tuple[BoundaryCondition, BoundaryCondition], ...]:
    """makes the boundary canonical ((low, high),) * ndim

    Accepts `None` for the reflecting default on every face.
    """
    if boundary is None:
        boundary = Neumann
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
            groups.setdefault(condition, []).append(2 * d + side)
    steps = [
        condition.define(sim, kernels, faces) for condition, faces in groups.items()
    ]

    def bc_step(u):
        for step in steps:
            step(u)
        return u

    return bc_step


def random_layer(
    sim: Simulation,
    width: int,
    contrast: float = 0.5,
    faces: Sequence[int] | None = None,
    correlation: int = 1,
    rng: int | np.random.Generator | None = None,
) -> cpt.NDArray:
    """Multiplicative random field over the `width` nodes inside a face, 1 elsewhere.

    A reflecting face behind a randomized layer scatters the incoming wave back
    incoherently instead of absorbing it, so the operator stays lossless and the
    time reversal `superposition_sensitivity` needs survives (Shen & Clapp 2015).

    Args:
        sim: the simulation whose padded grid the layer is built on.
        width: layer thickness in nodes, measured inward from the wall node.
        contrast: peak relative perturbation, reached at the wall and tapered to 0
            at the inner edge so the layer front is not a coherent reflector.
        faces: the `2 * axis + side` codes to line, or None for every face.
        correlation: scatterer size in nodes, 1 drawing one value per node.
        rng: seed or generator, so a driver reproduces its layer.

    Returns:
        a field over the padded grid, multiplying whatever `sim` reads the indicator
        as -- the impedance under `ScalarWave`, the wave speed under `AcousticWave`.
    """
    if not 0.0 <= contrast < 1.0:
        raise ValueError(f"contrast must lie in [0, 1) to stay positive: {contrast}")
    if width < 1:
        raise ValueError(f"width must be at least one node: {width}")
    if correlation < 1:
        raise ValueError(f"correlation must be at least one node: {correlation}")
    faces = tuple(range(2 * sim.ndim) if faces is None else faces)
    if any(f not in range(2 * sim.ndim) for f in faces):
        raise ValueError(
            f"face codes run to {2 * sim.ndim - 1} in {sim.ndim}D: {faces}"
        )
    for d in range(sim.ndim):
        sides = sum(2 * d + side in faces for side in (0, 1))
        if sides * width >= sim.Nx[d] - 2:
            raise ValueError(
                f"{sides} layer(s) of {width} nodes leave no interior on axis {d}: "
                f"Nx={sim.Nx[d]}"
            )

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
                # max, so a corner is perturbed once rather than by each of its faces
                taper = cp.maximum(taper, ramp)
    # the ghost ring is slaved to its mirror and the padding tail is never read
    taper *= interior

    # one draw for the whole grid, coarsened and repeated back up to `correlation`
    coarse = tuple((n + correlation - 1) // correlation for n in sim.Nx_padded)
    xi = np.random.default_rng(rng).uniform(-1.0, 1.0, size=coarse)
    for d in range(sim.ndim):
        xi = np.repeat(xi, correlation, axis=d)
    trim = tuple(slice(n) for n in sim.Nx_padded)
    return 1.0 + contrast * taper * cp.asarray(xi[trim], dtype=sim.dtype)
