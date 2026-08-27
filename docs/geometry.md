# Geometry

**Geometry** builds boolean region masks on the padded simulation grid, which a driver then turns into a material contrast

Every helper takes the `coords` that `grid_coords` returns and an optional `out` mask to accumulate into, created when omitted. So a shape is one call, and a composite geometry is a chain of calls sharing one `out` — there is no scene object and no set algebra beyond the union that accumulation gives

| member | signature | description |
|---|---|---|
| 2D shape | `ellipse(coords, center, radii, angle=0.0, out=None)` | interior of the ellipse with semi-axes `radii`, rotated by `angle` radians |
| 2D shape | `circle(coords, center, radius, out=None)` | interior of the circle, the isotropic `ellipse` |
| random field | `random_ellipses(coords, count, radii, bounds, angle=(0, pi), overlap=True, rng=None, attempts=100, out=None)` | `count` ellipses at uniformly random centers, semi-axes and orientations, optionally rejected until none overlap |
| graded row | `stacked_circles(coords, count, radius, span, center, axis=0, ratio=0.5, order="descending", out=None)` | a row of geometrically shrinking circles, evenly gapped along one axis and tangent to both ends of `span` |

Masks are boolean, so the material contrast is the driver's to choose: `cp.where(mask, GAMMA_VOID, 1.0)` for a two-phase indicator, or an arithmetic blend for a smoothed one

`random_ellipses` takes `rng` as a seed or a generator so a driver reproduces its geometry, and raises rather than silently placing fewer shapes once `attempts` draws in a row are rejected. `stacked_circles` raises when the requested circles do not fit into `span` — a defect row that quietly overlapped would make the recovered field impossible to score against

A staircased mask is only $O(h)$-consistent across resolutions, so a convergence study wants a finite transition width instead of a sharp `circle`, built by blending on the radius rather than thresholding it
