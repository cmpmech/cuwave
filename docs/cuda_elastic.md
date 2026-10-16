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

## pml helpers
Every pml kernel of this file runs on the slab of one lined axis `e` through `SLAB_OR_RETURN` ([cuda_scalar](cuda_scalar.md)). Its memory `aux` holds `2 * NDIM` rows: the strain memories $\psi_k$ of $\partial_eu_k$ at the strain point $\left(k,e\right)$, then the divergence memories $\zeta_c$ of the axis-$e$ part of $\nabla\cdot\sigma_c$. The `profile` rows are read at the node or at the half point $a+\tfrac12$, whichever the quantity sits on along `e`
### zeroed_axes
**aim**
the bitmask of axes a traction wall zeroes at a node, the set bits of `normal_strains` alone, so the pml stress condenses as the step does
### stress_row, Stress
**aim**
`stress_row(c, e)`: the Voigt row of $\sigma_{ce}$, normal on the diagonal, else its shear pair; `Stress`: $\sigma_{ce}$ `j` nodes along `e` from a node, the stress component the divergence of $c$ reads along `e`
### strain_term
**aim**
$\partial_eu_k$ at the strain point $\left(k,e\right)$, graded at the walls exactly as `normal_strains` and `shear_strain` take it, including the clamped fold
**input args**
`u`: any accessor of `(k, j)`, a `Field` forwards and a `Memory` in the adjoint; `k`, `e`: component and axis; `A`, `NN`: node indices and extents; `fe`: $1/\Delta x_e$; `clamped`: the face bitmask
### divergence_term
**aim**
the axis-$e$ part of the divergence of component $c$, as `fd_kernel` takes it, with the same graded taps and clamped fold
**input args**
`sig`: any accessor of `(c, j)`, a `Stress` forwards and a `Memory<Stress>` in the adjoint; `c`, `e`, `A`, `NN`, `fe`, `clamped`: as `strain_term`
### add_stress
**aim**
add the stress of the strain terms `t` of axis `e` onto the stress points, through the same condensed $\lambda$ and shear means as `stress_kernel`
**input args**
`sigma`: the stress scratch, added to; `gnode`, `gshear`, `lam`, `mu`, `clamped`: as `stress_kernel`; `idx`, `cs`, `e`, `A`, `NN`: geometry; `t`: one strain term per component, $\partial_eu_k$ or its memory

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

### pml_stress_kernel
**parallelization**
flat 1D launch over the slab of axis `e`, `PML_THREADS` per block; launched once per lined axis after `stress_kernel`
**aim**
advance the strain memories of axis `e` and add the stress they carry onto `sigma`
**input args**
`u1`: the field at $t-1$; `sigma`: the stresses `stress_kernel` wrote, added to; `gnode`, `gshear`, `lam`, `mu`, `clamped`, `cs`: as `stress_kernel`; `aux`: the memory of this axis; `profile`: its `(4, NN[e])` rows from `pml_profiles`; `e`: the lined axis; `lo, hi`: slab rows; `f0..f2`: $1/\Delta x_d$; `N0..N2`, `s0, s1`: geometry
**how?**
$$\psi_k\leftarrow b\,\psi_k+a\,\partial_eu_k,\qquad\sigma\mathrel{+}=\textrm{the stress of }\psi$$
- every lined axis adds onto the one scratch, so the update reads the stress stretched along all of them, which is what holds a corner together
### pml_force_kernel
**parallelization**
as `pml_stress_kernel`, launched once per lined axis after `fd_kernel`
**aim**
advance the divergence memories of axis `e` and add them to the update
**input args**
`u1`: unread, kept so one launch list serves this kernel and its adjoint; `u2`: the field at $t$, added to; `sigma`: the stretched stresses; `minv`, `clamped`, `cs`: as `fd_kernel`; `aux`, `profile`, `e`, `lo`, `hi`: as `pml_stress_kernel`; `f0..f2`: $\Delta t^2/\Delta x_d$; `N0..N2`, `s0, s1`: geometry
**how?**
$$\zeta_c\leftarrow b\,\zeta_c+a\,\partial_e\sigma_{ce},\qquad u_c^t\mathrel{+}=m_c^{-1}\zeta_c$$
### pml_adjoint_stress_kernel
**parallelization**
as `pml_stress_kernel`, the slab grown `STENCIL_RADIUS` rows past each lined front, since the transposed taps reach that far; launched after `stress_kernel` has formed the stress of $\lambda$
**aim**
push the divergence memories back through the strain: the transpose of `pml_force_kernel`
**input args**
`l1`: the adjoint at $t+1$; `sigma`: the stress of $\lambda$, added to; `aux`: the adjoint memories, read only; the rest as `pml_stress_kernel`
**how?**
- `strain_term` on the `Memory` of $\lambda$ and the divergence memories, $a\left(\lambda_k+\eta_{\zeta,k}\right)$, then `add_stress`
### pml_adjoint_force_kernel
**parallelization**
as `pml_adjoint_stress_kernel`, launched after `fd_kernel`
**aim**
push the strain memories back through the divergence, the transpose of `pml_stress_kernel`, and advance the divergence memories
**input args**
`l1`: the adjoint at $t+1$; `l2`: the adjoint at $t$, added to; `sigma`, `minv`, `clamped`, `cs`, `aux`, `profile`, `e`, `lo`, `hi`: as `pml_force_kernel`
**how?**
$$\lambda_c^t\mathrel{+}=m_c^{-1}\,\partial_e\left(a\left(\sigma_{ce}+\eta_{\psi}\right)\right),\qquad\eta_{\zeta,c}\leftarrow b\left(\eta_{\zeta,c}+\lambda_c^{t+1}\right)$$
- the divergence memory is advanced only on the slab proper, where `t` is not -1, after the stress pass has read it
### pml_adjoint_memory_kernel
**parallelization**
over the slab proper, launched last in every adjoint step
**aim**
advance the strain memories once the update has read them
**input args**
`sigma`: the stretched stress of $\lambda$; `cs`, `aux`, `profile`, `e`, `lo`, `hi`, `f0..f2`, `N0..N2`, `s0, s1`: as above
**how?**
$$\eta_{\psi,k}\leftarrow b\left(\eta_{\psi,k}+\sigma_{ke}\right)$$

### excitation_kernel
From the shared prelude, so this file does not carry it; see [cuda_scalar](cuda_scalar.md). A vector source is `NDIM` entries of `lin_index`, one per component, at `d * cs + idx`, so the `atomicAdd` covers a source and a sensor landing on the same point and component

### get_signal_kernel
From the shared prelude, the component folded into `lin_index` the same way

### set_signal_kernel
From the shared prelude, assignment rather than `atomicAdd`, so it restores a recorded state

The `sigma` scratch is allocated once by `define_step` and reused every step: the two launches replace the single fused one because each stress value is read by up to $2r$ updates, so forming it once trades one round of global traffic against recomputing every strain per reader
