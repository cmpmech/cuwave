# CUDA elastic

The forward kernels of the staggered [elastic](elastic.md) equation: one pass forming the stresses on their staggered points, one pass gathering their divergence into the update

## compilation logic

| flag | set from | effect |
|---|---|---|
| `USE_FLOAT` | `precision` | `real_t` is `float` rather than `double` |
| `NDIM` | `len(Nx)` | selects the interior guard, the geometry macros and the loop bounds |
| `USE_DAMPING` | `damping is not None` | adds the damped update and its two extra arguments |
| `USE_PML` | `pml is not None` | folds the pml memory terms into both step kernels, adds their per-axis arguments and the `phase`, and orders the blocks along axis 0 by `BLOCK_ROW` |
| `STENCIL_RADIUS` | `space_order // 2` | taps per staggered difference, prepended by [stencils](stencils.md) |
| `STAG_COEFFS` | `staggered_weights` | the graded coefficient table, prepended alongside |
| `TILE0`, `TILE1`, `TILE2` | `FUSED_TILE`, by `ElasticWave.define_step` without a pml where the fields spill out of L2 and the ring at most doubles the tile | compiles `fused_kernel`, the tile one block steps |

`NPAIRS` is $\textrm{ndim}\left(\textrm{ndim}-1\right)/2$ and `NVOIGT` is `NDIM + NPAIRS`, the stress components held back to back in the `sigma` scratch. `SG_W(r, k)` reads coefficient $k$ of the radius-$r$ row from `__constant__` memory, collapsing to 1 at `STENCIL_RADIUS` 1. `PAIR_ROW(k, l)` maps an axis pair onto its Voigt row, `NDIM + 3 - k - l` in 3D and 2 in 2D

## macros

### INTERIOR_OR_RETURN
Declares `a0`, `a1`, `a2` and `idx` and returns on the ghost ring and the padding tail, exactly as in [cuda_scalar](cuda_scalar.md). The guard is the union over the point families; each family checks its own staggered axes against `NN[d] - 3` inside

### BLOCK_ROW
Under `USE_PML`, the block index along axis 0 taken alternately from either end, $0, n-1, 1, n-2, \ldots$, so the slab blocks along that axis, which do the most work, are scheduled in the first wave instead of trailing the last one. Without it, the identity

### PML_FOLD, FOLD_GEOM
Under `USE_PML`, the per-axis arguments appended to `stress_kernel` and `fd_kernel` and the arrays `M`, `P`, `LO`, `HI` they fill; empty otherwise

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

### deep_node, node_radius, half_radius, clamped_at
**aim**
the wall-free path: a node more than $2R$ nodes inside every wall reads no graded tap, no wall weight and no clamped fold
**input args**
`A`, `NN`: geometry; `DEEP`: the template switch
**how?**
- with `DEEP` set, `node_radius` and `half_radius` return `STENCIL_RADIUS` and `clamped_at` returns false, so the tap guards and the `SG_W` row are compile-time constants; unset, they are `rad_node`, `rad_half` and `clamped_face`
- every helper below that reads a radius or a wall takes `DEEP` as a template argument defaulting to false, and `stress_kernel` and `fd_kernel` branch on `deep_node` per thread, so only the warps that hold a near-wall node run both paths
- the deep path is the general one with the wall terms removed, so the two agree bitwise

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

## material helpers

### wall_weight
The cell weight $W$ of a node, halved on the walls of every axis outside the bitmask `skip`, which are the axes the point is staggered along

### node_gamma, shear_gamma
**aim**
the design on the normal and on the $\left(k,l\right)$ shear stress point, formed from `gamma` per read rather than stored
**how?**
$$W\gamma\qquad\textrm{and}\qquad w_s\gamma_s=w_s\,\frac{4}{\sum_{4\textrm{ nodes}}1/\gamma}$$
- the harmonic mean over the four nodes around the shear point, `TINY` guarding a void node as the Python averages do

### point_minv
**aim**
the inverse inertia of component `c` on its own point, zero where a clamped wall holds it
**how?**
$$m_c^{-1}=\frac{1}{\rho_0\,W\,\tfrac12\left(\gamma_i+\gamma_{i+e_c}\right)}$$
- the volume is left out, since the force leaves it out too; `ElasticWave.point_inverse_inertia` evaluates the same value on the host for the excitation weights

## step helpers

### stress_point
**aim**
every stress of the node at `idx`, zero off the point a family does not carry, as `stress_kernel` writes them; see there for the formulas

### nodal_force
**aim**
the stress divergence on component `c`'s point, as `fd_kernel` takes it; see there for the taps

## pml helpers
The pml terms are folded into `stress_kernel` and `fd_kernel`: a thread whose node lies on the slab of a lined axis `e` adds that axis' memory terms in registers, so a lined step costs no launch beyond the unlined one. The memory `aux` of axis `e` holds `2 * NDIM` rows: the strain memories $\psi_k$ of $\partial_eu_k$ at the strain point $\left(k,e\right)$, then the divergence memories $\zeta_c$ of the axis-$e$ part of $\nabla\cdot\sigma_c$. The adjoint adds a second block of `NDIM` strain rows at `2 * NDIM`: `fd_kernel` reads one block at the neighbours while settling the other at its own node, and the two swap every step. Its slab row `t` is `slab_index` of the node ([cuda_scalar](cuda_scalar.md)). The `profile` rows are read at the node or at the half point $a+\tfrac12$, whichever the quantity sits on along `e`
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
add the stress of the strain terms `t` of axis `e` onto the node's stresses, through the same condensed $\lambda$ and shear means as `stress_kernel`
**input args**
`sig`: the node's `NVOIGT` stresses in registers, added to; `gamma`, `lam`, `mu`, `clamped`: as `stress_kernel`; `idx`, `e`, `A`, `S`, `NN`: geometry; `t`: one strain term per component, $\partial_eu_k$ or its memory
### slab_offset, in_slab
**aim**
`slab_offset`: the node's position across axis `d`, the other axes flattened as `slab_node` numbers them; `in_slab`: whether row `a` lies in the `lo` or `hi` rows of axis `e`, grown `ext` rows past each front
### stress_memory
**aim**
axis `e`'s memory terms on the node's stresses, for `stress_kernel`
**input args**
`u`: the field the stress is formed from; `aux`, `profile`: the memory and the `(4, NN[e])` rows from `pml_profiles` of axis `e`; `lo, hi`: its slab rows; `phase`: 0 forwards, else the adjoint; `sig`: the node's stresses, added to; the rest as `stress_kernel`
**how?**
- forwards, on the slab: $\psi_k\leftarrow b\,\psi_k+a\,\partial_eu_k$, and `add_stress` adds the stress of $\psi$; every lined axis adds onto the same stresses, so the update reads the stress stretched along all of them, which is what holds a corner together
- in the adjoint, on the slab grown `STENCIL_RADIUS` rows, since the transposed taps reach that far: the divergence memories pushed back through the strain, `strain_term` on the `Memory` of $\lambda$ and $\eta_\zeta$, $a\left(\lambda_k+\eta_{\zeta,k}\right)$, then `add_stress`: the transpose of the forward divergence term
### force_memory
**aim**
axis `e`'s memory term on component `c`'s update, for `fd_kernel`, which adds it times $m_c^{-1}$
**input args**
`sigma`: the stretched stresses; `u1`: the field at $t-1$, read by the adjoint only; `c`: the component; the rest as `stress_memory`
**how?**
- forwards: $\zeta_c\leftarrow b\,\zeta_c+a\,\partial_e\sigma_{ce}$, returned
- in the adjoint, on the grown slab: the strain memories pushed back through the divergence, $\partial_e\left(a\left(\sigma_{ce}+\eta_\psi\right)\right)$ off the block `phase` names, the transpose of the forward strain term; then, on the slab proper, where `t` is not -1, the divergence memory $\eta_{\zeta,c}\leftarrow b\left(\eta_{\zeta,c}+\lambda_c^{t+1}\right)$, which the stress pass has already read
### settle_memory
**aim**
advance the adjoint's strain memories of axis `e`, for `fd_kernel` after its update
**how?**
$$\eta_{\psi,k}\leftarrow b\left(\eta_{\psi,k}+\sigma_{ke}\right)$$
read off the block `phase` names and written into the other, so no thread overwrites a memory a neighbour's `force_memory` still reads

## kernels

### stress_kernel
**parallelization**
one thread per grid point, writing every stress family whose point it carries
**input args**
`u1`: the field at $t-1$; `gamma`: the nodal design, which `node_gamma` and `shear_gamma` take onto the stress points; `lam`, `mu`: $\lambda$ (plane-reduced on the host) and $\mu$; `clamped`: the face bitmask; `cs`: component stride; `f0..f2`: $1/\Delta x_d$; `N0..N2`, `s0, s1`: geometry; under `USE_PML`, per axis `m`, `p`, `lo`, `hi`: its memory, profile and slab rows, 0 unlined, then `phase`: 0 forwards, 1 or 2 the adjoint reading its strain memory off row 0 or `2 * NDIM`
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
`u0`, `u1`: the fields at $t-2$ and $t-1$; `sigma`: the stresses `stress_kernel` just wrote; `gamma`, `rho`: the nodal design and $\rho_0$, from which `point_minv` forms the inverse inertia per component; `damping`: nodal $d$, with `USE_DAMPING` only; `dt`: $\Delta t$, with `USE_DAMPING` only; `clamped`, `cs`: as above; `f0..f2`: $\Delta t^2/\Delta x_d$; the pml arguments of `stress_kernel`, under `USE_PML`
**output args**
`u2`: the field at $t$, same layout as `u1`
**internal args**
`force`: the divergence per component; `ap, am`: the stress node of the plus and minus tap; `rp, rm`: that point's own graded radius, which decides whether the tap exists
**how?**
- the divergence is the exact transpose of the strain differences: every tap carries the coefficient of the stress point it reads, evaluated by the same `rad_node` / `rad_half`, with the clamped fold doubling the wall tap
$$f_c=\frac{1}{\Delta x_c}\sum_k c_k^{\left(r\left(a\right)\right)}\left(\sigma_{cc}^{i+k}-\sigma_{cc}^{i-k+1}\right)+\sum_{l\ne c}\frac{1}{\Delta x_l}\sum_j c_j^{\left(r\left(a\right)\right)}\left(\sigma_{cl}^{i+j-1}-\sigma_{cl}^{i-j}\right)$$
- the march is the three-term recursion of [cuda_scalar](cuda_scalar.md), component by component, `USE_DAMPING` dividing by $1+\beta$ with one damping field serving every component
- under `USE_PML`, `force_memory` of every lined axis adds onto the update, $u_c^t\mathrel{+}=m_c^{-1}\zeta_c$, and the adjoint then settles its strain memories with `settle_memory`

### fused_kernel
`stress_kernel` and `fd_kernel` in one launch, bitwise the pair; without a pml
**parallelization**
one block per tile of `TILE0` $\times$ `TILE1` ($\times$ `TILE2`) nodes, a thread each; the block first forms the stresses of the tile and of a ring `STENCIL_RADIUS` wide around it, then updates its own nodes
**input args**
`u0`, `u1`, `u2`, `gamma`, `rho`, `damping`, `dt`, `clamped`, `cs`, `f0..f2`: as `fd_kernel`; `lam`, `mu`: as `stress_kernel`; `e0..e2`: $1/\Delta x_d$, the factors `stress_kernel` takes
**internal args**
`ss`: the `NVOIGT` stresses of the tile and its ring in shared memory, zero where `stress_kernel` writes nothing (off the interior); `local`, `T`: the node's place and strides in it, which `nodal_force` reads through as it would the `sigma` scratch
**how?**
- the ring holds every stress the divergence of a tile node reaches, so a block needs nothing another block forms; the ring's stresses are formed twice, once by each block that reads them
- the stresses never travel through memory: a step reads $u^{t-1}$, $u^{t-2}$ and the design and writes $u^t$, against the pair's extra write and read of every stress, which halves the traffic where DRAM bounds the step. While the scratch fits in L2 the pair stays as fast or faster, and where the ring more than doubles the stresses a block forms (3D above order 2) their recomputation costs more than the traffic saved, so `define_step` keeps the pair there

### excitation_kernel
From the shared prelude, so this file does not carry it; see [cuda_scalar](cuda_scalar.md). A vector source is `NDIM` entries of `lin_index`, one per component, at `d * cs + idx`, so the `atomicAdd` covers a source and a sensor landing on the same point and component

### get_signal_kernel
From the shared prelude, the component folded into `lin_index` the same way

### set_signal_kernel
From the shared prelude, assignment rather than `atomicAdd`, so it restores a recorded state

The `sigma` scratch is allocated once by `define_step` and reused every step: the two launches replace a single fused one because each stress value is read by up to $2r$ updates, so forming it once trades one round of global traffic against recomputing every strain per reader. A tile that forms the stresses in shared memory pays that recomputation on its halo, which at the block sizes a register budget allows costs more than the scratch saves. The materials are the opposite case: averaging $\gamma$ per read costs a few loads the cache already holds, and spares the inertia and shear grids, so a 2D step holds the fields, `gamma` and the scratch alone
