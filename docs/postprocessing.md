# Postprocessing

**Postprocessing** draws the bare field figures a driver produces — the wave, the design over it, and the transducers that made it — so the drivers carry the physics and not the axes bookkeeping

`matplotlib` is imported here and nowhere else in the package, the way torch is confined to [nn](nn.md), and it stays an optional extra in `pyproject.toml`. Everything works in **node index** coordinates rather than physical ones: an axes is the grid at one pixel per node, so a marker, an outline and the field itself are placed by the same integers the kernels use

| member | signature | description |
|---|---|---|
| axes | `field_axes(resolution, dpi=100, pad=0.05, panels=1, gap=0.15)` | a borderless axes the size of the grid, ticks and spines cleared and the image run to the edge of the figure, returning one axes or a `panels`-long array |
| field | `show(ax, field=None, indicator=None, cmap="seismic", saturation=1.0, scale=None, indicator_cmap="binary", gray=1.0)` | the wave field colored symmetrically about zero, the design drawn over it where it is solid, or either one alone |
| marker size | `marker_points(ax, nodes=6.0)` | the size in points that spans `nodes` grid nodes on `ax` |
| transducers | `markers(ax, positions, dx=None, origin=0, nodes=6.0, color="silver", **style)` | dots at grid node indices, or at physical coordinates when `dx` is given |
| region | `outline(ax, low, high, origin=0, color="silver", linewidth=1.5)` | a rectangle around the nodes `low` to `high`, marking a target or a design region |
| between panels | `arrow(fig, left, right, color="gray", scale=20.0, inset=0.25)` | a horizontal arrow across the gap between two panels, in figure coordinates |
| write | `save(fig, path)` | writes the figure on a transparent background, so one run gives a light and a dark variant of the same plot |

A driver plotting an interior slice passes `origin=1`, since node 1 is the origin of the domain and becomes cell 0 of the plotted array; a driver plotting the whole logical grid leaves `origin` at 0. The half-cell offset from a cell corner to its center is applied inside `markers` and `outline`, so a dot lands on its node rather than on the corner below it

`show` takes both layers because that is the figure the applications want: the design is what was optimized or recovered, and the wave running through it is what the design was scored on. Given `field` and `indicator` together the design is drawn over the wave as a flat `gray`, transparent everywhere else; given `indicator` alone it is drawn as a $[0,1]$ field in `cmap`, which is the plain design plot a [fwi](fwi.md) driver wants next to its truth

**A marker size in points does not track the grid**, which is why `marker_points` exists: the same `markersize=4` is a fifth of a 20-node domain and invisible on a 2000-node one. Sizing in nodes instead,
$$\textrm{points}=n_\textrm{nodes}\cdot 72\cdot\frac{w}{x_\textrm{max}-x_\textrm{min}}$$
with the axes width `w` in inches and its node-index limits, holds a dot at a fixed fraction of the grid at any resolution and any dpi. `nodes` is still a per-figure choice like a colormap — a very wide figure is displayed smaller and wants a larger dot — but the guarantee is that the choice means the same thing in every figure

`scale` defaults to the peak amplitude of `field`, which a point source dominates: its injection node is orders of magnitude above the wave it launched, so the propagating field washes out. Pass a percentile of the field instead, or dim the default with `saturation`, whenever the figure has a source in it
