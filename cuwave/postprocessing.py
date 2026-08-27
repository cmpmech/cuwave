"""Matplotlib helpers for the bare field figures the drivers produce

`matplotlib` is imported here and nowhere else in the package, the way torch is confined
to `nn.py`, so the solver keeps its cupy-only dependency. Everything works in node index
coordinates: an axes is the grid at one pixel per node, and a marker is sized in nodes,
so the same call gives the same figure at any resolution and any dpi
"""

import cupy.typing as cpt
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch


# -------------------------------------- helpers --------------------------------------
def _host(field: cpt.NDArray | npt.NDArray) -> npt.NDArray:
    """`field` on the host, whichever array module it came from"""
    return np.asarray(field.get() if hasattr(field, "get") else field)


def _panel_width(ax: Axes) -> float:
    """Width of `ax` in inches, without asking for a renderer"""
    return ax.get_position().width * ax.figure.get_figwidth()


# -------------------------------------- figures --------------------------------------
def field_axes(
    resolution: tuple[int, ...],
    dpi: int = 100,
    pad: float = 0.05,
    panels: int = 1,
    gap: float = 0.15,
) -> tuple[Figure, Axes]:
    """Borderless axes the size of the grid, padded so a marker on the edge survives.

    Args:
        resolution: (nx, ny) nodes of the plotted array, which sets the figure size.
        dpi: dots per inch, the figure being `resolution` pixels wide at 100.
        pad: margin around the grid, as a fraction of the longest axis.
        panels: side-by-side panels, each one `resolution` wide.
        gap: spacing between panels, as a fraction of one panel's width.

    Returns:
        (fig, ax), `ax` a single axes for one panel and a `panels`-long array beyond.
    """
    width = resolution[0] / 100 * (panels + gap * (panels - 1))
    fig, axes = plt.subplots(1, panels, figsize=(width, resolution[1] / 100), dpi=dpi)
    margin = pad * max(resolution)
    for ax in np.atleast_1d(axes):
        ax.set_xlim(-margin, resolution[0] + margin)
        ax.set_ylim(-margin, resolution[1] + margin)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=gap)
    return fig, axes


def show(
    ax: Axes,
    field: cpt.NDArray | npt.NDArray | None = None,
    indicator: cpt.NDArray | npt.NDArray | None = None,
    cmap: str = "seismic",
    saturation: float = 1.0,
    scale: float | None = None,
    indicator_cmap: str = "binary",
    gray: float = 1.0,
) -> Axes:
    """Draw a wave field, a design, or the design over the field.

    Args:
        ax: the axes from `field_axes`, which fixes the node index coordinates.
        field: (nx, ny) wave field, colored symmetrically about zero.
        indicator: (nx, ny) design, drawn over `field` where it is solid and left
            transparent elsewhere, or alone as a [0, 1] field in `cmap`.
        cmap: colormap of `field`, or of `indicator` when that is drawn alone.
        saturation: fraction of `scale` the colormap runs to, below 1 to clip the peak.
        scale: the amplitude `field` saturates at, defaulting to its own peak.
        indicator_cmap: colormap of the overlay, only reached together with `field`.
        gray: the constant the overlay is drawn at, 1 black and 0 white in `binary`.
    """
    if field is not None:
        values = _host(field)
        scale = float(np.max(np.abs(values))) if scale is None else scale
        limit = scale * saturation
        ax.pcolormesh(values.T, cmap=cmap, vmin=-limit, vmax=limit)
    if indicator is not None and field is None:
        ax.pcolormesh(_host(indicator).T, cmap=cmap, vmin=0, vmax=1)
    elif indicator is not None:
        solid = np.where(_host(indicator), gray, np.nan)
        ax.pcolormesh(solid.T, cmap=indicator_cmap, vmin=0, vmax=1)
    return ax


def save(fig: Figure, path: str) -> None:
    """Write `fig` to `path` on a transparent background, as the docs figures want"""
    fig.savefig(path, transparent=True)


# ------------------------------------- annotation ------------------------------------
def marker_points(ax: Axes, nodes: float = 6.0) -> float:
    """Marker size in points spanning `nodes` grid nodes, which points alone do not"""
    low, high = ax.get_xlim()
    return nodes * 72.0 * _panel_width(ax) / (high - low)


def markers(
    ax: Axes,
    positions: cpt.NDArray | npt.ArrayLike,
    dx: tuple[float, ...] | None = None,
    origin: int = 0,
    nodes: float = 6.0,
    color: str = "silver",
    **style,
) -> None:
    """Dots at grid `positions`, sized in nodes so they match across figures.

    Args:
        ax: the axes from `field_axes`, whose limits fix the points-per-node scale.
        positions: (ndim, count) grid node indices, or (count, ndim) physical
            coordinates when `dx` is given.
        dx: grid spacing, which turns physical coordinates into node indices.
        origin: node index of the plotted array's first cell, 1 for an interior slice.
        nodes: diameter of a dot in grid nodes, held the same figure to figure.
        color: fill and edge color, gray reading over both a light and a dark field.
        **style: passed to `plot`, e.g. `zorder` or `alpha`.
    """
    index = np.atleast_2d(_host(positions)).astype(float)
    if dx is not None:
        index = (index / np.asarray(dx)).T + 1.0
    # the pcolormesh cell of node i is centered half a node past its corner
    ax.plot(
        index[0] - origin + 0.5,
        index[1] - origin + 0.5,
        "o",
        markersize=marker_points(ax, nodes),
        markerfacecolor=color,
        markeredgecolor=color,
        linestyle="none",
        clip_on=False,
        zorder=3,
        **style,
    )


def outline(
    ax: Axes,
    low: tuple[int, ...],
    high: tuple[int, ...],
    origin: int = 0,
    color: str = "silver",
    linewidth: float = 1.5,
) -> None:
    """Rectangle around the nodes `low` to `high`, marking a target or design region"""
    ax.add_patch(
        plt.Rectangle(
            (low[0] - origin, low[1] - origin),
            high[0] - low[0] + 1,
            high[1] - low[1] + 1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )


def arrow(
    fig: Figure,
    left: Axes,
    right: Axes,
    color: str = "gray",
    scale: float = 20.0,
    inset: float = 0.25,
) -> None:
    """Horizontal arrow across the gap between two panels, in figure coordinates"""
    a, b = left.get_position(), right.get_position()
    span = b.x0 - a.x1
    y = 0.5 * (a.y0 + a.y1)
    fig.add_artist(
        FancyArrowPatch(
            (a.x1 + inset * span, y),
            (b.x0 - inset * span, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=scale,
            color=color,
            linewidth=0.1 * scale,
        )
    )
