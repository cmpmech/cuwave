# CUDA elastic

The forward kernels of the staggered [elastic](elastic.md) equation: one pass forming the stresses on their staggered points, one pass gathering their divergence into the update

## compilation logic

| flag | set from | effect |
|---|---|---|
| `USE_FLOAT` | `precision` | `real_t` is `float` rather than `double` |
| `NDIM` | `len(Nx)` | selects the interior guard, the geometry macros and the loop bounds |
| `USE_DAMPING` | `damping is not None` | adds the damped update and its two extra arguments |
| `STENCIL_RADIUS` | `space_order // 2` | taps per staggered difference, prepended by [stencils](stencils.md) |
| `STAG_COEFFS` | `staggered_weights` | the graded coefficient table, prepended alongside |

`NPAIRS` is $\textrm{ndim}\left(\textrm{ndim}-1\right)/2$ and `NVOIGT` is `NDIM + NPAIRS`, the stress components held back to back in the `sigma` scratch. `SG_W(r, k)` reads coefficient $k$ of the radius-$r$ row from `__constant__` memory, collapsing to 1 at `STENCIL_RADIUS` 1. `PAIR_ROW(k, l)` maps an axis pair onto its Voigt row, `NDIM + 3 - k - l` in 3D and 2 in 2D

## macros

### INTERIOR_OR_RETURN
Declares `a0`, `a1`, `a2` and `idx` and returns on the ghost ring and the padding tail, exactly as in [cuda_scalar](cuda_scalar.md). The guard is the union over the point families; each family checks its own staggered axes against `NN[d] - 3` inside

### AXIS_GEOM, AXIS_FACTORS
Declare the compile-time arrays the family loops index: `A` the per-axis grid index, `S` the per-axis stride (last is 1), `NN` the per-axis logical extent, `F` the per-axis factor `f0..f2`

## device functions

### rad_node, rad_half
**aim**
the graded radius of a staggered difference, keyed off the **stress point** so the forward and the transpose read the same matrix entry
**input args**
`a`: the point's index along the derivative axis; `N`: logical extent
**how?**
- a node strain reads unknowns at $a-r\ldots a+r-1$, valid over $1\ldots N-3$, so
$$r_\textrm{node}\left(a\right)=\min\left(R,\,a-1,\,N-2-a\right)$$
which is zero on the wall itself; a half-point derivative reads nodes at $a+1-r\ldots a+r$, valid over $1\ldots N-2$, so
$$r_\textrm{half}\left(a\right)=\min\left(R,\,a,\,N-2-a\right)$$

### clamped_face
Reads bit $2d+\textrm{side}$ of the `clamped` mask, the faces `Simulation.boundary` marks `Clamped`

### normal_strains
**aim**
the node's normal strains, one per axis, and which axes graded to a zero row
**input args**
`u`: a displacement field, `NDIM` components of `cs` entries back to back; `idx`, `A`, `S`, `NN`, `F`: geometry, `F` holding $1/\Delta x_d$
**output args**
`eps`: $\varepsilon_{dd}$ per axis; the return value: the bitmask of zeroed axes
**how?**
- in the interior, the graded staggered difference of the component along its own axis
$$\varepsilon_{dd}\big|_a=\frac{1}{\Delta x_d}\sum_{k\le r}c_k^{(r)}\left(u_d^{a+k-1}-u_d^{a-k}\right)$$
- on a clamped wall the strain folds antisymmetrically about the held wall value, $\varepsilon=\pm2u_d/\Delta x_d$ from the single unknown half a node inside
- on a traction wall the row is zero and the axis is reported for condensation

### condensed_lame
**aim**
condense each zeroed axis out of the normal coupling
**input args**
`lam`, `mu`: the Lamé scalars; `zeroed`: the bitmask from `normal_strains`
**how?**
- setting $\sigma_{dd}=0$ and eliminating $\varepsilon_{dd}$ is the plane stress reduction, applied once per zeroed axis
$$\lambda\leftarrow\frac{2\lambda\mu}{\lambda+2\mu}$$
so a surface-tangential stiffness is the statically condensed one, and an edge or corner condenses twice or three times

### shear_strain
**aim**
the engineering shear strain of the $\left(k,l\right)$ pair at its own staggered point
**input args**
`u`, `idx`, `cs`, `k`, `l`, `A`, `S`, `NN`, `F`: as above
**how?**
- both derivatives land on the point without averaging, each at its own graded radius
$$2\,\varepsilon_{kl}=\frac{1}{\Delta x_l}\sum_{j\le r_1}c_j\left(u_k^{a+j}-u_k^{a+1-j}\right)+\frac{1}{\Delta x_k}\sum_{j\le r_2}c_j\left(u_l^{a+j}-u_l^{a+1-j}\right)$$

## kernels

### stress_kernel
**parallelization**
one thread per grid point, writing every stress family whose point it carries
**input args**
`u1`: the field at $t-1$; `gnode`: $W\gamma$ on the nodes; `gshear`: `NPAIRS` fields of $w_s\gamma_s$, the harmonic shear means, `NDIM >= 2` only; `lam`, `mu`: $\lambda$ (plane-reduced on the host) and $\mu$; `clamped`: the face bitmask; `cs`: component stride; `f0..f2`: $1/\Delta x_d$; `N0..N2`, `s0, s1`: geometry
**output args**
`sigma`: `NVOIGT` fields, the normal stresses first and the pairs in Voigt order, assigned rather than accumulated; points no family writes stay at the zero the scratch was allocated with
**how?**
- the normal family couples through the (possibly condensed) $\lambda$, a zeroed axis writing an explicit zero
$$\sigma_{dd}=W\gamma\left(2\mu\,\varepsilon_{dd}+\lambda_\textrm{eff}\,\textrm{tr}\,\boldsymbol{\varepsilon}\right)$$
- each shear point is one multiply on its own strain
$$\sigma_{kl}=w_s\gamma_s\,\mu\left(2\,\varepsilon_{kl}\right)$$

### fd_kernel
**parallelization**
one thread per grid point, updating every component whose unknown it carries
**input args**
`u0`, `u1`: the fields at $t-2$ and $t-1$; `sigma`: the stresses `stress_kernel` just wrote; `minv`: `NDIM` fields of $1/\left(\rho_0W\bar{\gamma}\right)$, one per component at its own points, the volume deliberately left out since the force below leaves it out too; `damping`: nodal $d$, with `USE_DAMPING` only; `dt`: $\Delta t$, with `USE_DAMPING` only; `clamped`, `cs`: as above; `f0..f2`: $\Delta t^2/\Delta x_d$
**output args**
`u2`: the field at $t$, same layout as `u1`
**internal args**
`force`: the divergence per component; `ap, am`: the stress node of the plus and minus tap; `rp, rm`: that point's own graded radius, which decides whether the tap exists
**how?**
- the divergence is the exact transpose of the strain differences: every tap carries the coefficient of the stress point it reads, evaluated by the same `rad_node` / `rad_half`, with the clamped fold doubling the wall tap
$$f_c=\frac{1}{\Delta x_c}\sum_k c_k^{\left(r\left(a\right)\right)}\left(\sigma_{cc}^{i+k}-\sigma_{cc}^{i-k+1}\right)+\sum_{l\ne c}\frac{1}{\Delta x_l}\sum_j c_j^{\left(r\left(a\right)\right)}\left(\sigma_{cl}^{i+j-1}-\sigma_{cl}^{i-j}\right)$$
- the march is the three-term recursion of [cuda_scalar](cuda_scalar.md), component by component, `USE_DAMPING` dividing by $1+\beta$ with one damping field serving every component

### excitation_kernel
Byte-identical to [cuda_scalar](cuda_scalar.md). A vector source is `NDIM` entries of `lin_index`, one per component, at `d * cs + idx`, so the `atomicAdd` covers a source and a sensor landing on the same point and component

### get_signal_kernel
Byte-identical to [cuda_scalar](cuda_scalar.md), the component folded into `lin_index` the same way

### set_signal_kernel
Byte-identical to [cuda_scalar](cuda_scalar.md), assignment rather than `atomicAdd`, so it restores a recorded state

The `sigma` scratch is allocated once by `define_step` and reused every step: the two launches replace the single fused one because each stress value is read by up to $2r$ updates, so forming it once trades one round of global traffic against recomputing every strain per reader
