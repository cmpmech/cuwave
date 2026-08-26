"""Boolean region masks on the padded simulation grid

Every helper takes the ``coords`` returned by :func:`cuwave.wave.grid_coords` and an
optional ``out`` mask to accumulate into: it is created when omitted
"""

import math

import cupy as cp
import numpy as np


# -------------------------------------- helpers --------------------------------------
def _accumulate(mask, out):
    if out is None:
        return mask
    out |= mask
    return out


def _empty(coords, out):
    return cp.zeros(coords[0].shape, dtype=bool) if out is None else out


# ----------------------------------------- 1D ----------------------------------------


# ----------------------------------------- 2D ----------------------------------------
def ellipse(coords, center, radii, angle=0.0, out=None):
    """Interior of the ellipse with semi-axes ``radii``, rotated by ``angle`` radians"""
    dx = coords[0] - center[0]
    dy = coords[1] - center[1]
    if angle:
        cos, sin = math.cos(angle), math.sin(angle)
        dx, dy = cos * dx + sin * dy, cos * dy - sin * dx
    return _accumulate((dx / radii[0]) ** 2 + (dy / radii[1]) ** 2 < 1.0, out)


def circle(coords, center, radius, out=None):
    """Interior of the circle of radius ``radius``"""
    return ellipse(coords, center, (radius, radius), out=out)


def random_ellipses(
    coords,
    count,
    radii,
    bounds,
    angle=(0.0, math.pi),
    overlap=True,
    rng=None,
    attempts=100,
    out=None,
):
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
    coords,
    count,
    radius,
    span,
    center,
    axis=0,
    ratio=0.5,
    order="descending",
    out=None,
):
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


# ----------------------------------------- 3D ----------------------------------------
