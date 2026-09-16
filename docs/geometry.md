# Geometry

**Geometry** builds boolean region masks on the padded simulation grid, which a driver then turns into a material contrast

Every helper takes the `coords` that `grid_coords` returns and an optional `out` mask to accumulate into, created when omitted. So a shape is one call, and a composite geometry is a chain of calls sharing one `out`: there is no scene object and no set algebra beyond the union that accumulation gives

| member | signature | description |
|---|---|---|
| 2D shape | `ellipse(coords, center, radii, angle=0.0, out=None)` | interior of the ellipse with semi-axes `radii`, rotated by `angle` radians |
| 2D shape | `circle(coords, center, radius, out=None)` | interior of the circle, the isotropic `ellipse` |
| point cloud | `circles(coords, centers, radius, out=None)` | union of equal-radius circles, one per row of `centers`, in any dimension |
| 2D shape | `box(coords, center, sizes, out=None)` | interior of the axis-aligned box with side lengths `sizes`, its boundary included |
| 2D shape | `rectangle(coords, center, sizes, angle=0.0, out=None)` | interior of the rectangle with side lengths `sizes`, rotated by `angle` radians, the orientable `box` |
| random field | `random_ellipses(coords, count, radii, bounds, angle=(0, pi), overlap=True, rng=None, attempts=100, out=None)` | `count` ellipses at uniformly random centers, semi-axes and orientations, optionally rejected until none overlap |
| graded row | `stacked_circles(coords, count, radius, span, center, axis=0, ratio=0.5, order="descending", out=None)` | a row of geometrically shrinking circles, evenly gapped along one axis and tangent to both ends of `span` |

| nodes | `nodes(mask)` | the `(ndim, count)` grid indices `mask` selects, the layout [sensitivity](sensitivity.md) and `simulate` take |

`nodes` is the one member that consumes a mask rather than building one, and it is what makes a region usable as a sensor array: a target box is marked by coordinate like any other shape and then handed to the solver as the nodes that tile it, with no separate index arithmetic to keep in step

`circles` is the mask a driver freezes rather than fills: taking the transducer coordinates as `centers` marks the neighbourhood of every source and receiver, and zeroing the gradient there holds those nodes at the intact value. The gradient close to a point source is dominated by the singularity of the source itself rather than by the material, so an unconstrained inversion paints a ring of spurious damage around each transducer and spends its iterations on it, which `examples/fwi/fwi_2D_adam_mask.py` shows against the same setup without the mask

`rectangle` and `ellipse` are the two ways to draw a crack or a notch, and they part company at the tips: an ellipse tapers to a point, so a chain of them pinches to nothing wherever two segments meet, where rotated rectangles keep their full width across the junction. A defect that pinches leaks the wave it was meant to reflect, so a jointed geometry wants `rectangle` and a single smooth flaw wants `ellipse`

Masks are boolean, so the material contrast is the driver's to choose: `cp.where(mask, GAMMA_VOID, 1.0)` for a two-phase indicator, or an arithmetic blend for a smoothed one

`random_ellipses` takes `rng` as a seed or a generator so a driver reproduces its geometry, and raises rather than silently placing fewer shapes once `attempts` draws in a row are rejected. `stacked_circles` raises when the requested circles do not fit into `span`: a defect row that quietly overlapped would make the recovered field impossible to score against

A staircased mask is only $O(h)$-consistent across resolutions, so a convergence study wants a finite transition width instead of a sharp `circle`, built by blending on the radius rather than thresholding it
