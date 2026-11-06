# CUDA elastic

The forward kernels of the staggered [elastic](elastic.md) equation: one pass forming the stresses on their staggered points, one pass gathering their divergence into the update. Compiled with `GRADIENT`, the same two passes step the adjoint and add its gradient densities on the way

## compilation logic

| flag | set from | effect |
|---|---|---|
| `USE_FLOAT` | `precision` | `real_t` is `float` rather than `double` |
| `NDIM` | `len(Nx)` | selects the interior guard, the geometry macros and the loop bounds |
| `USE_DAMPING` | `damping is not None` | adds the damped update and its two extra arguments |
| `USE_PML` | `pml is not None` | folds the pml memory terms into both step kernels, adds their per-axis arguments and the `phase`, and orders the blocks along axis 0 by `BLOCK_ROW` |
| `STENCIL_RADIUS` | `space_order // 2` | taps per staggered difference, prepended by [stencils](stencils.md) |
| `STAG_COEFFS` | `staggered_weights` | the coefficient table one row per radius, prepended alongside, of which this file reads the full row |
| `NODE_CLOSURE`, `HALF_CLOSURE`, `NODE_SPAN`, `HALF_SPAN`, `NODE_ROWS_C`, `HALF_ROWS_C` | `staggered_pair` for `stencil`, orders 2 to 8 | the low wall's rows of the staggered SBP pair, how many of each and how far they reach, prepended alongside |
| `TILE0`, `TILE1`, `TILE2` | `FUSED_TILE`, by `ElasticWave.define_step` without a pml where the fields spill out of L2 and the ring at most doubles the tile | compiles `fused_kernel`, the tile one block steps |
| `STEPS`, `TILE0`, `TILE1`, `ROWS` | `BLOCKED` or `LARGE_BLOCKED`, by `ElasticWave.define_blocked_step` in 2D without a pml, on a small grid or past L2 | compiles `blocked_kernel` instead of `fused_kernel`: the steps a launch advances, the tile and the thread rows sharing it |
| `GRADIENT` | `ElasticWave.define_adjoint_gradient` | `stress_kernel`, `fd_kernel` and `fused_kernel` add the gradient densities of the adjoint `u1` against the forward `v1`, their accumulators appended to the arguments |
| `RELAXATION` | `relaxation.mechanisms`, never with a pml, `TILE` or `STEPS` | `stress_kernel` relaxes through that many memory variables per stress, the stresses after the stiffness, under `GRADIENT` the strains before it; appends `memory` and `coef` |

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

### STIFFNESS_GRADIENT, MASS_GRADIENT, SHEAR_ROWS
Under `GRADIENT`, the arguments appended to `stress_kernel` (`g_normal`, `g_shear` from 2D on, `v1`) and to `fd_kernel` (`g_mass`, `v1`, `mf`) ahead of the pml ones, and `SHEAR_ROWS` the shear accumulator a call passes on; empty otherwise

### RELAXATION_ARGS
Under `RELAXATION`, `memory` and `coef` appended to `stress_kernel` after the pml arguments, which a relaxation never comes with; empty otherwise

## device functions

### rad_half
The graded radius of a half-point difference, $\min\left(R,\,a,\,N-2-a\right)$ at the half point `a`, which the [maxwell](maxwell.md) pml terms take; the elastic step and its pml close their walls by the rows below

### clamped_face
Reads bit $2d+\textrm{side}$ of the `clamped` mask, the faces `Simulation.boundary` marks `Clamped`

### Line, node_rows, half_rows
**aim**
`Line`: a field along an axis, `u(j)` the value `j` points from `idx`; `node_rows`, `half_rows`: the closure row of the derivative onto node or half point `lo` points from the low wall, or mirrored and negated `hi` points from the high one
**input args**
`u`: any accessor of `j`, a `Line` in the step and an `Along` in the pml terms; `lo`, `hi`: the point's place from either wall
**how?**
- `NODE_ROWS[lo][m]` weighs the half point `m` from the wall (the wall point left out, since it is never read), `HALF_ROWS[lo][i]` the node `i`; the high wall reads row `hi` over the points from its own side, the high corner being the low one turned by half a turn
- every loop has a compile-time bound and a runtime offset, so it unrolls; the tables are `__device__` rather than `__constant__`, since along the fast axis a warp reads them at as many rows as it holds wall points

### node_derivative, half_derivative, node_divergence, half_divergence
**aim**
the staggered SBP pair of [stencils](stencils.md) (`staggered_pair`) along a line: the derivative $D_+$ onto node `a` of a half-point field, $D_-$ onto the half point `a` (at $a+\tfrac12$) of a nodal one, and the divergences the update takes, $D_-$ of a nodal stress and $D_+$ of a half-point one
**input args**
`u` or `sg`: the field along the line; `a`, `N`: the point's index along the axis and the logical extent; `DEEP`: the template switch
**how?**
- past the closure, the full staggered difference with `SG_W` at radius $R$
$$D_+u\big|_a=\sum_{k=1}^{R}c_k\left(u^{a+k-1}-u^{a-k}\right)\qquad D_-u\big|_a=\sum_{k=1}^{R}c_k\left(u^{a+k}-u^{a+1-k}\right)$$
- within `NODE_CLOSURE` nodes or `HALF_CLOSURE` half points of a wall, `node_rows` or `half_rows`
- the divergences are the other derivative, since the pair sums by parts, $H_+D_++\left(H_-D_-\right)^\top=B$: the update's $-M^{-1}D_+^\top W\sigma$ is $D_-\sigma$ with the norms of $W$ and $M$ cancelling, and $-M^{-1}D_-^\top W\sigma$ is $D_+\sigma$, the stress at the half-grid wall point read as zero. So no kernel reads a norm, and the update reaches no further than the strains
- `node_divergence` and `half_divergence` keep the interior taps in the order the deep path has always rounded them; with `DEEP` set the closure branches compile away, which is what `deep_node` buys

### deep_node, clamped_at
**aim**
the wall-free path: a node more than `CLOSURE_BAND` nodes inside every wall, the larger of the two row counts, reads no closure row and no clamped wall
**input args**
`A`, `NN`: geometry; `DEEP`: the template switch
**how?**
- with `DEEP` set, the derivatives take their interior branch alone and `clamped_at` returns false; unset, every helper checks its own axis' place against either wall
- every helper below that reads a wall takes `DEEP` as a template argument defaulting to false, and every kernel branches on `deep_node` per thread, so only the warps that hold a node within the band of a wall run both paths
- the deep path is the general one with the wall terms removed, rounded alike, so the two agree bitwise on a node they both reach and a kernel may take either there

### normal_strains
**aim**
the node's normal strains, one per axis, and which axes a traction wall zeroes
**input args**
`u`: a displacement field, `NDIM` components of `cs` entries back to back; `idx`, `A`, `S`, `NN`, `F`: geometry, `F` holding $1/\Delta x_d$
**output args**
`eps`: $\varepsilon_{dd}$ per axis; the return value: the bitmask of zeroed axes
**how?**
- `node_derivative` of the component along its own axis
$$\varepsilon_{dd}\big|_a=\frac{1}{\Delta x_d}\left(D_+u_d\right)_a$$
- on a clamped wall the closure's wall row reads the half-grid wall point, held at zero, as nothing
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
- both derivatives land on the point without averaging, each a `half_derivative` along its own axis
$$2\,\varepsilon_{kl}=\frac{1}{\Delta x_l}\left(D_-u_k\right)_{a_l}+\frac{1}{\Delta x_k}\left(D_-u_l\right)_{a_k}$$
- the sum is rounded by `MUL_RN` and `FMA_RN` on either path, so every kernel and both paths round it alike

## material helpers

### shear_gamma
**aim**
the design on the $\left(k,l\right)$ shear stress point, formed from `gamma` per read rather than stored, the normal stress taking the nodal value itself
**how?**
$$\gamma_s=\frac{4}{\sum_{4\textrm{ nodes}}1/\gamma}$$
- the harmonic mean over the four nodes around the shear point, `TINY` guarding a void node as the Python averages do

### point_minv
**aim**
the inverse inertia of component `c` on its own point, zero where a clamped wall holds it
**how?**
$$m_c^{-1}=\frac{1}{\rho_0\,\tfrac12\left(\gamma_i+\gamma_{i+e_c}\right)}$$
- the volume is left out, since the force leaves it out too, and so is the norm, which cancels against the stresses'; `ElasticWave.point_inverse_inertia` is this over the norm, the inverse mass a point load divides by

## step helpers

### strain_point
**aim**
every strain of the node at `idx`, the normal ones of `normal_strains` and the engineering shear ones of `shear_strain`, zero off the point a family does not carry
**output args**
`eps`: `NVOIGT` strains in Voigt order; the return value: the bitmask of zeroed axes

### stress_of
**aim**
the stresses of the strains `eps` at the node at `idx`, through the condensed $\lambda$ and the point means of the design; see `stress_kernel` for the formulas
**input args**
`eps`, `zeroed`: as `strain_point` returns them; `gamma`, `lam`, `mu`: as `stress_kernel`; `idx`, `A`, `S`, `NN`: geometry
**output args**
`sig`: `NVOIGT` stresses, zero off their points

### stress_point
`strain_point` then `stress_of`, every stress of the node as `stress_kernel` writes it without a relaxation

### relax
**aim**
relax the node's `NVOIGT` rows `s` through their memory variables, settling each in place
**input args**
`memory`: `RELAXATION * NVOIGT` grids, the variable of mechanism $l$ and Voigt row $v$ at row `l * NVOIGT + v`; `coef`: $c_0$, then $a_l, h_l, b_l$ per mechanism; `idx`, `cs`: node and component stride
**how?**
- the trapezoidal update centred on the step, the rows read before the memory moves
$$s\leftarrow c_0\,s-\sum_lh_l\,\zeta_l,\qquad\zeta_l\leftarrow a_l\,\zeta_l+b_l\,s,\qquad c_0=1-\frac12\sum_lb_l,\quad h_l=\frac{1+a_l}{2}$$
which is $\sigma_U-\tfrac12\sum_l\left(\zeta_l^{n+\frac12}+\zeta_l^{n-\frac12}\right)$ of [elastic](elastic.md)
- forwards `s` are the stresses of `stress_point` and $\zeta$ stress memories; under `GRADIENT` `s` are the adjoint's strains of `strain_point` and $\zeta$ strain memories, equivalent since the stiffness is pointwise and constant in time

### nodal_force
**aim**
the stress divergence on component `c`'s point, as `fd_kernel` takes it: `node_divergence` of its normal stress along its own axis, then `half_divergence` of each shear stress along the others, summed through `MUL_RN` and `FMA_RN` so the pair, the fused and the blocked step round alike

### wall_stresses
`stress_point` of a ring point within a wall's band into the fused kernel's shared memory, out of line, so the registers the general path takes do not cost the ring's deep points their occupancy

## gradient helpers

Under `GRADIENT` only. `normal_form` is the one of [cuda_elastic_sensitivity](cuda_elastic_sensitivity.md), copied, since each file is its own module

### stiffness_density
**aim**
subtract the stiffness densities of the adjoint against the forward field at the node and its shear points, as `gradient_kernel` does
**input args**
`eps`: the adjoint's `NVOIGT` strains, under `RELAXATION` already relaxed by `relax`; `v1`: the forward field they pair with; `idx`, `cs`, `A`, `S`, `NN`: geometry; `F`: $1/\Delta x_d$, the strain factors; `lam`, `mu`, `clamped`: as `stress_kernel`; `DEEP`: as `deep_node`
**output args**
`g_normal`, `g_shear`: the normal and shear accumulators, decremented
**how?**
- the strains of `v1` on the path `deep_node` picks, which rounds as the general one, so the densities are bit for bit those of `gradient_kernel`
$$g_n\mathrel{-}=2\mu\,\boldsymbol{\varepsilon}\left(\lambda\right):\boldsymbol{\varepsilon}\left(u\right)+\lambda_\textrm{eff}\,\textrm{tr}\,\boldsymbol{\varepsilon}\left(\lambda\right)\textrm{tr}\,\boldsymbol{\varepsilon}\left(u\right)\qquad g_{kl}\mathrel{-}=\mu\,2\varepsilon_{kl}\left(\lambda\right)\,2\varepsilon_{kl}\left(u\right)$$
- under a relaxation the density of the relaxed recursion, whose stiffness acts on the relaxed strain

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
$\partial_eu_k$ at the strain point $\left(k,e\right)$, the step's own `node_derivative` or `half_derivative` along an `Along`, zero on a traction wall's normal row
**input args**
`u`: any accessor of `(k, j)`, a `Field` forwards and a `Memory` in the adjoint, which `Along` turns into a line of component `k`; `k`, `e`: component and axis; `A`, `NN`: node indices and extents; `fe`: $1/\Delta x_e$; `clamped`: the face bitmask
### divergence_term
**aim**
the axis-$e$ part of the divergence of component $c$, the step's `node_divergence` or `half_divergence`. In the weighted variables $W\sigma$ the pair is $D$ and $-D^\top$, each other's transpose, which is what the adjoint terms below rest on; the norms cancel out of the kernels as in the step
**input args**
`sig`: any accessor of `(c, j)`, a `Stress` forwards and a `Memory<Stress>` in the adjoint; `c`, `e`, `A`, `NN`, `fe`: as `strain_term`
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
`u1`: the field at $t-1$; `gamma`: the nodal design, the normal stress point's own value and `shear_gamma`'s mean on a shear point; `lam`, `mu`: $\lambda$ (plane-reduced on the host) and $\mu$; `clamped`: the face bitmask; `cs`: component stride; `f0..f2`: $1/\Delta x_d$; `N0..N2`, `s0, s1`: geometry; under `GRADIENT`, `v1`: the forward field `u1` pairs with; under `USE_PML`, per axis `m`, `p`, `lo`, `hi`: its memory, profile and slab rows, 0 unlined, then `phase`: 0 forwards, 1 or 2 the adjoint reading its strain memory off row 0 or `2 * NDIM`; under `RELAXATION`, `memory`, `coef`: as `relax`, the memory settled in place
**output args**
`sigma`: `NVOIGT` fields, the normal stresses first and the pairs in Voigt order, assigned rather than accumulated; points no family writes stay at the zero the scratch was allocated with; under `GRADIENT`, `g_normal`, `g_shear`: the stiffness densities of `stiffness_density`, accumulated with `-=`
**how?**
- the normal family couples through the (possibly condensed) $\lambda$, a zeroed axis writing an explicit zero
$$\sigma_{dd}=\gamma\left(2\mu\,\varepsilon_{dd}+\lambda_\textrm{eff}\,\textrm{tr}\,\boldsymbol{\varepsilon}\right)$$
- each shear point is one multiply on its own strain
$$\sigma_{kl}=\gamma_s\,\mu\left(2\,\varepsilon_{kl}\right)$$
- no norm weighs either, since the update's divergence is the other derivative of the pair, against which they cancel
- under `RELAXATION`, `relax` on those stresses before the store
- under `GRADIENT`, the adjoint's strains of `strain_point` once, for `stiffness_density` and `stress_of` alike; under a relaxation `relax` on those strains in between

### fd_kernel
**parallelization**
one thread per grid point, updating every component whose unknown it carries
**input args**
`u0`, `u1`: the fields at $t-2$ and $t-1$; `sigma`: the stresses `stress_kernel` just wrote; `gamma`, `rho`: the nodal design and $\rho_0$, from which `point_minv` forms the inverse inertia per component; `damping`: nodal $d$, with `USE_DAMPING` only; `dt`: $\Delta t$, with `USE_DAMPING` only; `clamped`, `cs`: as above; `f0..f2`: $\Delta t^2/\Delta x_d$; under `GRADIENT`, `v1`: the forward field `u1` pairs with, and `mf`: $1/\Delta t^2$; the pml arguments of `stress_kernel`, under `USE_PML`
**output args**
`u2`: the field at $t$, same layout as `u1`, which the adjoint passes as `u0` again; under `GRADIENT`, `g_mass`: the inertia density, accumulated with `-=`
**internal args**
`force`: the divergence per component, from `nodal_force`
**how?**
- the divergence is the transpose of the strain differences in the norms, which `node_divergence` and `half_divergence` form as the other derivative of the pair, so the inertia carries no norm either
$$f_c=\frac{1}{\Delta x_c}\left(D_-\sigma_{cc}\right)_{a_c}+\sum_{l\ne c}\frac{1}{\Delta x_l}\left(D_+\sigma_{cl}\right)_{a_l}$$
- the march is the three-term recursion of [cuda_scalar](cuda_scalar.md), component by component, `USE_DAMPING` dividing by $1+\beta$ with one damping field serving every component
- under `USE_PML`, `force_memory` of every lined axis adds onto the update, $u_c^t\mathrel{+}=m_c^{-1}\zeta_c$, and the adjoint then settles its strain memories with `settle_memory`
- under `GRADIENT`, the inertia density summed by parts in time, the forward field against the second difference of the adjoint, formed before the store so the `u0` it reads is not yet overwritten
$$g_{m,c}\mathrel{-}=\frac{1}{\Delta t^2}v_c\left(u_c^t-2u_c^{t-1}+u_c^{t-2}\right)$$

### fused_kernel
`stress_kernel` and `fd_kernel` in one launch, bitwise the pair; without a pml
**parallelization**
one block per tile of `TILE0` $\times$ `TILE1` ($\times$ `TILE2`) nodes, a thread each; the block first forms the stresses of the tile and of a ring `STENCIL_RADIUS` wide around it, then updates its own nodes. `tile_origin` puts each axis' last tile flush with its end and returns -1 past it, and `tile_end` stops a tile where the next begins, so the last tiles overlap their neighbour and a node is stepped once
**input args**
`u0`, `u1`, `u2`, `gamma`, `rho`, `damping`, `dt`, `clamped`, `cs`, `f0..f2`: as `fd_kernel`; `lam`, `mu`: as `stress_kernel`; `e0..e2`: $1/\Delta x_d$, the factors `stress_kernel` takes; under `GRADIENT`, after the geometry, `g_mass`, `g_normal`, `g_shear`, `v1`, `mf`: as the pair, both densities formed at the block's own nodes, the adjoint's strains read off global memory with `e0..e2`
**internal args**
`ss`: the `NVOIGT` stresses of the tile and its ring in shared memory, zero where `stress_kernel` writes nothing (off the interior); `local`, `T`: the node's place and strides in it, which `nodal_force` reads through as it would the `sigma` scratch
**how?**
- the ring holds every stress the divergence of a tile node reaches, so a block needs nothing another block forms; the ring's stresses are formed twice, once by each block that reads them. A closure row reaches further than the ring, towards its wall, which the flush last tile keeps inside the block that holds the wall, as long as a tile is wider than the rows reach (`closure_reach`, which `define_step` checks)
- ring points within a band take `wall_stresses`, and the launch bounds the registers to three blocks a multiprocessor to radius 2 and two past it, which the deep path fits without spilling
- the stresses never travel through memory: a step reads $u^{t-1}$, $u^{t-2}$ and the design and writes $u^t$, against the pair's extra write and read of every stress, which halves the traffic where DRAM bounds the step. While the scratch fits in L2 the pair stays as fast or faster, and where the ring more than doubles the stresses a block forms (3D above order 2) their recomputation costs more than the traffic saved, so `define_step` keeps the pair there

### blocked_kernel
`STEPS` steps of `stress_kernel`, `fd_kernel`, the excitation and the record between one load and one store of a tile, bitwise the launches one at a time; 2D, float32, without a pml
**parallelization**
a 2D launch of `(TILE1, ROWS)` threads per block, one block per tile of `TILE0` $\times$ `TILE1` nodes; a thread owns `NODES` $=$ `TILE0 / ROWS` consecutive nodes of its column. Tiles overlap by a halo of `HALO` $=$ `STEPS` $\cdot(2R-1)$ nodes each side, the reach of a strain and a divergence, and write back only their core, `C0` $\times$ `C1`
**input args**
`u0, u1`: $u^{t-1}, u^t$, both components; `v0, v1`: the arrays receiving $u^{t+S-1}, u^{t+S}$; `gamma, lam, mu`: as `stress_kernel`; `rho` and under `-DUSE_DAMPING` `damping, dt`: as `fd_kernel`; `signal, weights, num_sources, record, num_sensors, t0`: as the scalar `blocked_kernel`; `source_starts, source_entries`, `sensor_starts, sensor_entries`: per tile the sources its nodes hold, halo included, and the sensors its core records, as CSR rows of `component node | point << 16`, the node counted once per component before it; `clamped, cs`: as `fd_kernel`; `e0, e1`: $1/\Delta x_d$; `f0, N0, f1, N1, s0`: as `fd_kernel`; `o0, o1`: the first core's first node per axis, 1 or before
**internal args**
`su, ss, sg`: both components of $u$, the three stresses and the design in shared memory, each with a guard ring $R$ wide, read by `stress_point` and `nodal_force` through the tile strides `T` exactly as they read the global arrays (`point_minv` reads the global design once, at load); `up, uc, mi, beta, tag`: per owned node and component in registers; `inner, moving`: bit masks of the owned interior nodes and of the components the update moves, leaving out a node within a wall's closure band when the tile does not hold that wall (`reads_held`), whose rows would read past the tile
**how?**
- per step the stresses of the tile go to shared memory, a barrier, then the update of both components, a barrier; a component the update skips (its own axis' staggered ghost) and every non-interior node keep their value of two steps back, as the plain buffers do
- the edge nodes of a tile read the guard ring and go wrong, one reach deeper every step; after `STEPS` steps the error stops short of the core. A band node `reads_held` leaves out goes wrong at once, so `define_blocked_step` shifts the cores (`o0, o1`) until no tile's edge falls inside a band it does not hold the wall of, and refuses where no shift does
- the damped update is rounded as nvcc compiles `fd_kernel`'s, $\Delta t\,\beta$ fused into $1\mp\beta$ with $\beta$ held over $\Delta t$, and the source term rounded apart as `excitation_kernel` adds it
- on a small grid a block steps a node a thread, since a step there is bound by the latency of one tile's stresses and update, which several nodes a thread would serialize; past L2 two nodes a thread on a wider tile trade that latency for the halo's traffic

### excitation_kernel
From the shared prelude, so this file does not carry it; see [cuda_scalar](cuda_scalar.md). A vector source is `NDIM` entries of `lin_index`, one per component, at `d * cs + idx`, so the `atomicAdd` covers a source and a sensor landing on the same point and component

### get_signal_kernel
From the shared prelude, the component folded into `lin_index` the same way

### set_signal_kernel
From the shared prelude, assignment rather than `atomicAdd`, so it restores a recorded state

The `sigma` scratch is allocated once by `define_step` and reused every step: the two launches replace a single fused one because each stress value is read by up to $2r$ updates, so forming it once trades one round of global traffic against recomputing every strain per reader. A tile that forms the stresses in shared memory pays that recomputation on its halo, which at the block sizes a register budget allows costs more than the scratch saves. The materials are the opposite case: averaging $\gamma$ per read costs a few loads the cache already holds, and spares the inertia and shear grids, so a 2D step holds the fields, `gamma` and the scratch alone
