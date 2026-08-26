"""Boundary conditions, one per (axis, side).

A `BoundaryCondition` is a declarative marker: it names a kernel and nothing
more, so a setup can place one per face in `Simulation.boundary` long before
the module is compiled. `define_boundary` turns the markers into the launch
closures once `compile_kernels` has run, which is why `simulate` needs no
separate preparation phase.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, repr=False)
class BoundaryCondition:
    """Names a kernel in wave.cu. Frozen, so it can key the grouping below."""

    kernel: str

    def __repr__(self):
        # so that printing a Simulation shows the wall, not the kernel name
        return self.kernel.removesuffix("_kernel")

    def define(self, sim, kernels, faces):
        """closure applying this condition on `faces`, the codes 2 * axis + side"""
        kernel = kernels.get_function(self.kernel)
        threads = 256

        # one thread per boundary node: each face holds only the interior of the
        # remaining axes, so edges and corners are excluded
        inner = [n - 2 for n in sim.Nx]
        num_boundary = sum(
            int(np.prod(inner[:d] + inner[d + 1 :])) for d in (f // 2 for f in faces)
        )
        grid = ((num_boundary + threads - 1) // threads,)

        # the faces travel as a bitmask, so the kernel can keep its axis loop at
        # compile time and no device array has to be allocated or kept alive
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
Dirichlet = BoundaryCondition("dirichlet_kernel")


def canonical_boundary(boundary, ndim):
    """makes the boundary canonical ((low, high),) * ndim

    Accepts `None` for the default reflecting wall everywhere.
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


def define_boundary(sim, kernels):
    """one launch per distinct condition, so the default wall stays one launch."""
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
