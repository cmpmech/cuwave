"""Boolean region masks on the padded simulation grid

Every helper takes the `coords` returned by `grid_coords` and an optional `out` mask to
accumulate into: it is created when omitted
"""

import math
from collections.abc import Sequence

import cupy as cp
import cupy.typing as cpt
import numpy as np


# -------------------------------------- helpers --------------------------------------
def _accumulate(
    mask: cpt.NDArray[cp.bool_], out: cpt.NDArray[cp.bool_] | None
) -> cpt.NDArray[cp.bool_]:
    if out is None:
        return mask
    out |= mask
    return out


def _empty(
    coords: Sequence[cpt.NDArray], out: cpt.NDArray[cp.bool_] | None
) -> cpt.NDArray[cp.bool_]:
    return cp.zeros(coords[0].shape, dtype=bool) if out is None else out


def ellipse(
    coords: Sequence[cpt.NDArray],
    center: Sequence[float],
    radii: Sequence[float],
    angle: float = 0.0,
    out: cpt.NDArray[cp.bool_] | None = None,
) -> cpt.NDArray[cp.bool_]:
    """Interior of the ellipse with semi-axes `radii`, rotated by `angle` radians"""
    dx = coords[0] - center[0]
    dy = coords[1] - center[1]
    if angle:
        cos, sin = math.cos(angle), math.sin(angle)
        dx, dy = cos * dx + sin * dy, cos * dy - sin * dx
    return _accumulate((dx / radii[0]) ** 2 + (dy / radii[1]) ** 2 < 1.0, out)


def circle(
    coords: Sequence[cpt.NDArray],
    center: Sequence[float],
    radius: float,
    out: cpt.NDArray[cp.bool_] | None = None,
) -> cpt.NDArray[cp.bool_]:
    """Interior of the circle of radius `radius`"""
    return ellipse(coords, center, (radius, radius), out=out)


def box(
    coords: Sequence[cpt.NDArray],
    center: Sequence[float],
    sizes: Sequence[float],
    out: cpt.NDArray[cp.bool_] | None = None,
) -> cpt.NDArray[cp.bool_]:
    """Interior of the axis-aligned box with side lengths `sizes`, boundary included"""
    inside = None
    for x, c, size in zip(coords, center, sizes):
        slab = cp.abs(x - c) <= 0.5 * size
        inside = slab if inside is None else inside & slab
    return _accumulate(inside, out)


def random_ellipses(
    coords: Sequence[cpt.NDArray],
    count: int,
    radii: tuple[float, float],
    bounds: Sequence[tuple[float, float]],
    angle: tuple[float, float] = (0.0, math.pi),
    overlap: bool = True,
    rng: int | np.random.Generator | None = None,
    attempts: int = 100,
    out: cpt.NDArray[cp.bool_] | None = None,
) -> cpt.NDArray[cp.bool_]:
    """`count` ellipses at uniformly random centers, semi-axes, and orientations.

    Args:
        coords: the grid the mask is built on.
        count: how many ellipses to place.
        radii: (min, max) a semi-axis is drawn from, independently per axis.
        bounds: one (low, high) per axis, the box centers are drawn from.
        angle: (min, max) rotation in radians.
        overlap: when false, reject a center whose circumscribed circle meets an
            earlier one, and raise once `attempts` draws in a row are rejected.
        rng: seed or generator, so a driver reproduces its geometry.
        out: mask to accumulate into, created when omitted.

    Returns:
        the accumulated mask, true inside the ellipses.
    """
    rng = np.random.default_rng(rng)
    out = _empty(coords, out)
    placed = []  # (center, circumscribed radius) of the ellipses accepted so far
    for i in range(count):
        for _ in range(attempts):
            center = tuple(rng.uniform(low, high) for low, high in bounds)
            semi = tuple(rng.uniform(*radii) for _ in bounds)
            bound = max(semi)
            if overlap or all(
                math.dist(center, other) >= bound + radius for other, radius in placed
            ):
                break
        else:
            raise RuntimeError(
                f"placed only {i} of {count} ellipses without overlap "
                f"({attempts} attempts for the next one)"
            )
        placed.append((center, bound))
        ellipse(coords, center, semi, rng.uniform(*angle), out)
    return out


def stacked_circles(
    coords: Sequence[cpt.NDArray],
    count: int,
    radius: float,
    span: tuple[float, float],
    center: float | Sequence[float],
    axis: int = 0,
    ratio: float = 0.5,
    order: str | None = "descending",
    out: cpt.NDArray[cp.bool_] | None = None,
) -> cpt.NDArray[cp.bool_]:
    """A row of geometrically shrinking circles, evenly gapped along one axis.

    Args:
        coords: the grid the mask is built on.
        count: how many circles to place.
        radius: the largest radius, which `ratio` shrinks from.
        span: (start, end) along `axis` the row is fitted into, tangent to both ends.
        center: the coordinate on each of the other axes.
        axis: the axis the circles are stacked along.
        ratio: factor between consecutive radii.
        order: `descending` or `ascending` along the axis, or None for equal radii.
        out: mask to accumulate into, created when omitted.

    Returns:
        the accumulated mask, true inside the circles.
    """
    if order not in (None, "ascending", "descending"):
        raise ValueError("order must be 'descending', 'ascending' or None")
    if order is None:
        radii = [radius] * count
    else:
        radii = [radius * ratio**i for i in range(count)]
    if order == "ascending":
        radii.reverse()

    length = span[1] - span[0]
    gap = (length - 2.0 * sum(radii)) / (count - 1) if count > 1 else 0.0
    if gap < 0.0:
        raise ValueError(
            f"{count} circles of radius {radius} and ratio {ratio} do not fit "
            f"into a span of {length}"
        )

    others = np.atleast_1d(center).tolist()  # the nonstacking axes, in order
    if len(others) != len(coords) - 1:
        raise ValueError(f"center needs {len(coords) - 1} coordinate(s), not {others}")

    out = _empty(coords, out)
    position = span[0] + radii[0]
    for i, r in enumerate(radii):
        if i:
            position += radii[i - 1] + gap + r
        origin = others.copy()
        origin.insert(axis % len(coords), position)
        circle(coords, origin, r, out)
    return out


def nodes(mask: cpt.NDArray[cp.bool_]) -> cpt.NDArray[cp.int32]:
    """The (ndim, count) grid indices `mask` selects, as the kernels take them"""
    return cp.stack(cp.nonzero(mask)).astype(cp.int32)
