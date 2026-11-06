# CUDA scalar sensitivity

## what is differentiated
One forward step of [cuda_scalar](cuda_scalar.md), written as a residual instead of an update
$$C^t=\frac{m}{\Delta t^2}\left(u^t-2u^{t-1}+u^{t-2}\right)-\nabla\cdot(k\nabla u^{t-1})-b^t=0$$
Making $J-\sum_t\lambda^t C^t$ stationary in $u$ defines the **adjoint field** $\lambda$: the same three-term recursion, run backwards in time, driven at the sensors by $\partial J/\partial u$ in place of the source
$$\lambda^{t}=2\lambda^{t+1}-\lambda^{t+2}+\frac{\Delta t^2}{m}\left(\nabla\cdot(k\nabla\lambda^{t+1})+\frac{\partial J}{\partial u^{t}}\right)$$
where the divergence acting on $\lambda$ is the **transpose** of the forward one under the cell weights $W$, not the forward one again: above order 2 the wide flux under a varying $k$ makes the two differ. `adjoint_kernel` steps $\lambda$ with that transpose, and `adjoint_gradient_kernel` steps it and accumulates both gradients in one pass; the boundary kernels are the forward ones. What is left over is $dJ/d\theta=-\sum_t\lambda^t\,\partial C^t/\partial\theta$, one sum per material field, and those two sums are all this file computes
$$\frac{dJ}{dm}\bigg|_i=-\frac{1}{\Delta t^2}\sum_t\lambda_i^t\left(u_i^t-2u_i^{t-1}+u_i^{t-2}\right)$$
$$\frac{dJ}{dk}\bigg|_i=\sum_t\frac{\partial}{\partial k_i}\left[\lambda^t\cdot\nabla\cdot(k\nabla u^{t-1})\right]$$
Both are exact derivatives of the **discretization**, not of the PDE. And both differentiate the residual rather than the update, which is why neither carries the $\Delta t^2$ that the forward step folds into $f_d$: the mass term gets it back as `inv_dt2`, and the stiffness term runs on $F_d=f_d/\Delta t^2$, i.e. $2c_0^2/h_d^2$ (`ScalarWave`) or $2/h_d^2$ (`AcousticWave`)
The prelude of [cuda_scalar](cuda_scalar.md) is prepended here too, so `real_t`, the coefficient accessors and the mirrored taps are not repeated

## the transposed operator
One axis, nodes $0..N-1$ with the ghosts at both ends and the wall nodes $1$ and $N-2$. The forward step reads every tap past a wall node off its mirror, $\tilde u_{1-j}=\sigma\tilde u_{1+j}$, so every node takes the full radius $R$
$$(Au)_i=f\left(g_{i+\frac{1}{2}}D_i^+-g_{i-\frac{1}{2}}D_i^-\right),\qquad D_i^+=\sum_{k=1}^{R}w_k\left(\tilde u_{i+k}-\tilde u_{i-k+1}\right),\qquad D_i^-=\sum_{k=1}^{R}w_k\left(\tilde u_{i+k-1}-\tilde u_{i-k}\right)$$
with the mirrored field $\tilde u$, the sign $\sigma$ ($+1$ on a Neumann face, $-1$ on a Dirichlet one), the cell coefficients $w_k$ (`OP_W(R, k)`) and the cell stiffness $g$, which the even stiffness mirror (`wave.mirror_ghosts`, read past the ghost through its mirror) makes symmetric about the wall node. Its transpose under the cell weights $W$ is a gather over the rows that read node $a$
$$\left(W^{-1}A^\top W\lambda\right)_a=f\sum_{k=1}^{R}w_k\left(g_{a+k-\frac{1}{2}}\left(\tilde\lambda_{a+k}-\tilde\lambda_{a+k-1}\right)-g_{a-k+\frac{1}{2}}\left(\tilde\lambda_{a-k+1}-\tilde\lambda_{a-k}\right)\right)$$
with $\tilde\lambda$ mirrored as $\tilde u$ is: the **wide divergence of compact cell fluxes** where the forward step is the compact divergence of wide ones, which is not $A$ again wherever $g$ varies under a wide stencil
- near a wall, the reads past it fold onto their mirrors in $A$'s columns, and the wall node owns half a cell ($w_1=\tfrac12$). Mirroring $\lambda$ absorbs both: the half weight and the wall node's fold onto itself cancel, so the formula above holds at every node and the wall path is the deep one with every load through `reflected`
- a Dirichlet wall node is held at zero: the forward step writes it zero and the refill zeroes it, so its row of $A$ vanishes and its column meets only zeros; $\lambda$ is read as zero there (`reflected`), by its neighbours and by its own gradient, so an adjoint load a sensor puts there goes nowhere
- the stiffness gradient is the bracket of `stiffness_gradient_taps` on mirrored $u$, $\lambda$ and $k$
$$G_a=\frac{\partial g_{a+\frac{1}{2}}}{\partial k_a}D_a^+\left(\tilde\lambda_{a+1}-\tilde\lambda_a\right)+\frac{\partial g_{a-\frac{1}{2}}}{\partial k_a}D_a^-\left(\tilde\lambda_a-\tilde\lambda_{a-1}\right)$$
and the stiffness ghost, slaved to its mirror, lands on that mirror through the same reads
- a ghost is only ever read along its own axis, so all of this holds axis by axis and no corner term appears. Every read of $\lambda$ past a wall node goes to its mirror, so whatever the boundary kernels write into $\lambda$'s ghosts is never read

## split launch
The three transposed kernels (`gradient_kernel`, `adjoint_kernel`, `adjoint_gradient_kernel`) run the deep formula wherever a node's whole gather is deep and the general one elsewhere. A warp runs both when its lanes disagree, so the launch is arranged so that they never do along the fast axis
### WALL_REACH, NEAR_WALL, WALL_COLUMNS
- `WALL_REACH`: $R+2$: a node closer to a ghost than this reads a wall node or a tap past it
- `NEAR_WALL(a, N)`: whether node `a` of an axis of logical extent `N` is that close to either ghost
- `WALL_COLUMNS`: $2\left(\textrm{WALL\_REACH}-1\right)$, the stepped near-wall nodes of one fast-axis row
### DEEP_SPAN, wall_blocks, wall_thread, wall_column, deep_column
**aim**
map a thread of the grid `scalar.split_grid_block` sizes to a node
**how?**
- the leading `wall_blocks()` blocks of each block row along $x$ take the near-wall columns; `wall_thread()` counts their threads across **all** of them, so every row's `WALL_COLUMNS` nodes are packed densely into the first few blocks and the surplus blocks retire at once
- `wall_column(c, X)` turns slot `c` into the node $1+c$ on the low side or $X-2\,\textrm{WALL\_REACH}+1+c$ on the high one, and returns $-1$ for a high slot the low side already holds, which happens only on a grid narrower than $2\,\textrm{WALL\_REACH}$
- `deep_column(X)`: the remaining blocks take the columns `WALL_REACH` $..$ $X-1-$`WALL_REACH`, one per thread, as `grid_block` would
- `DEEP_SPAN(lo, n, N)`: whether the block's rows `lo .. lo + n - 1` of a slower axis are all deep. It depends on the block indices alone, so the branch it feeds is uniform per block
### SPLIT_OR_RETURN
`INTERIOR_OR_RETURN` for the split grid: declares `a0, a1, a2` and `idx` from the helpers above, returns past the interior, and declares `deep`, true on a block that holds only deep columns and deep rows
### INTERIOR_OR_RETURN, AXIS_OFFSETS
`fd_kernel`'s opening lines, as macros rather than a device function because they declare the axis indices and `return` early; `frechet_kernel` and `superposed_kernel` keep them
- `INTERIOR_OR_RETURN`: maps the thread to `a0, a1, a2` with the fastest axis on `x` (as `grid_block` does on the host), retires the ghost ring and the padding tail, and forms the flat index `idx`
- `AXIS_OFFSETS`: `o0, o1, o2`, the neighbour stride per axis, for the kernel that never reaches further than one node

## gradient helpers
### stiffness_gradient_taps
**aim**
compute one axis of $\partial\left[\lambda\cdot\nabla\cdot(k\nabla u)\right]/\partial k_i$ at node $i$, the taps from accessors
**input args**
`u`, `l`, `k`: accessors returning $u$, $\lambda$ and $k$ the given number of nodes along the axis; `uc`, `lc`, `sc`: the three at the node; `factor`: $F_d$, the same per-axis factor `fd_kernel` multiplies in but **without** the $\Delta t^2$
**internal args**
`sp, sm`: $k_{i\pm1}$; `dgp, dgm`: the two cell stiffnesses differentiated with respect to `sc`; `Dp, Dm`: the two fluxes $D_i^+, D_i^-$ at the full radius $R$
**how?**
- $\lambda\cdot\nabla\cdot(k\nabla u)$ is a sum over neighboring cells, not nodes: the cell at $i+\frac{1}{2}$ carries one stiffness $g^+$ and one flux $D_i^+$, and enters node $i$ with a $+$ and node $i+1$ with a $-$
$$\lambda\cdot\nabla\cdot(k\nabla u)=-\sum_d F_d\sum_{\textrm{cells}}g^+D_i^+\left(\lambda_{i+1}-\lambda_i\right)$$
- $k_i$ sits in exactly two of those cells, so the derivative stays local
$$\frac{\partial}{\partial k_i}\left[\lambda\cdot\nabla\cdot(k\nabla u)\right]=-\sum_d F_d\left[\frac{\partial g^+}{\partial k_i}D_i^+\left(\lambda_{i+1}-\lambda_i\right)+\frac{\partial g^-}{\partial k_i}D_i^-\left(\lambda_i-\lambda_{i-1}\right)\right]$$
- and the derivative of the (halved) harmonic cell mean is as cheap as the mean itself
$$g^\pm=\frac{k_ik_{i\pm1}}{k_i+k_{i\pm1}}\qquad\Longrightarrow\qquad\frac{\partial g^\pm}{\partial k_i}=\left(\frac{k_{i\pm1}}{k_i+k_{i\pm1}}\right)^2$$
- this is $G_a$ of the transposed operator. The function returns the bracket alone, so the outer minus is applied once by the caller (`g_stiff -= g`)
### transposed_divergence_taps
**aim**
compute one axis of $W^{-1}A^\top W\lambda$ at a node, without its factor $f_d$, the taps from accessors
**input args**
`l`, `k`: accessors as in `stiffness_gradient_taps`; `lc`, `sc`: $\lambda$ and $k$ at the node
**internal args**
`sh, sl, lh, ll`: $k$ and $\lambda$ at the inner end of the two cells $j$ out, rolled outward one node per tap; `sh1, sl1, lh1, ll1`: the same at the outer end
**how?**
- the gather of the transposed operator, $\sum_jw_j\left(g_{a+j-\frac{1}{2}}\left(\lambda_{a+j}-\lambda_{a+j-1}\right)-g_{a-j+\frac{1}{2}}\left(\lambda_{a-j+1}-\lambda_{a-j}\right)\right)$
- it forms $2R$ cell stiffnesses where `flux_divergence_axis` forms two, and reads $k$ at all $2R+1$ taps: that, not the arithmetic, is what a transposed step costs over a forward one
### stiffness_gradient_axis, transposed_divergence_axis
the two above at a deep node, every tap read in place from `u1`, `l1` and `stiff` at `idx` with stride `s`
### reflected
a field at node `a + j` of the line through `idx`, of an axis of `N` nodes: past the wall node its mirror, odd past a face of `odd`, and zero on a Dirichlet wall node. Unlike the forward `mirrored` it reflects the stored ghost too, which the adjoint never refills
### mirrored_gradient, mirrored_divergence
the two above at a node `NEAR_WALL` along the axis, `a` of `N`: every tap read through `reflected`, $u$ and $\lambda$ with the axis' Dirichlet bits `odd`, $k$ evenly. The caller passes `lc` as zero on a held node, the one read `reflected` does not make
### flux_divergence_axis
with `flux_divergence_taps`, the forward stencil of [cuda_scalar](cuda_scalar.md) byte for byte, since each `.cu` is its own compilation unit; `superposed_kernel` steps with it and sums the axes in `fd_kernel`'s pinned order, so the field it steps is bitwise the forward one
## pml helpers
The forward memory terms of `pml_kernel` in [cuda_scalar](cuda_scalar.md) are a linear recursion local to the slab, so their transpose is mechanical: the adjoint memories $\eta^\pm,\eta_\zeta$ run backwards with the same $b$, and what the forward kernel read off $u^n$ through $D^\pm$ the adjoint scatters back onto $\lambda$ as a gather over the taps
### plus_tap, minus_tap
**aim**
$\partial D^+/\partial u_{a+q}$ and $\partial D^-/\partial u_{a+q}$ at the full radius: the weight a cell difference gives the node `q` rows off, zero past its taps
### pml_tap
**aim**
what the memories of slab thread `i`, on layer row `ai`, give back to the node `q` rows off it
**input args**
`aux`: the adjoint memories $\eta^+,\eta^-,\eta_\zeta$, three rows of `size`; `stiff`: array of $k$; `profile`, `f`, `N`: as `pml_kernel`; `o`: stride along the axis; `i`: the slab thread; `iidx`: its flat index; `ai`: its row; `q`: the row offset to the reading node
**internal args**
`gp, gm`: harmonic cell stiffnesses of row `ai`; `ez`: $a_i f_d\eta_\zeta$; `cp, cm`: the coefficients on $D^+$ and $D^-$
**how?**
$$c^\pm=a_{i\pm\frac12}\eta^\pm\pm g^\pm\left(1+a_{i\pm\frac12}\right)a_if_d\eta_\zeta,\qquad\textrm{tap}=c^+\,\partial_qD^++c^-\,\partial_qD^-$$
## kernels
### gradient_kernel
**parallelization**
the split launch: one node per thread, the near-wall columns of the fast axis packed into the leading blocks (`SPLIT_OR_RETURN`)
**aim**
add one time step's contribution to both gradient sums
**input args**
`g_mass`, `g_stiff`: the two accumulators over the padded grid, added into; `u0, u1, u2`: the forward triplet $u^{t-2}, u^{t-1}, u^{t}$; `l1`: $\lambda^{t}$, named after `u1` because the stiffness term pairs the two; `stiff`: array of $k$; `inv_dt2`: $1/\Delta t^2$; `F0, F1, F2`: per-axis factors $f_d/\Delta t^2$; `N0, N1, N2`, `s0, s1`: grid dimensions and strides as in `fd_kernel`
**internal args**
`uc, lc, sc`: central entries of $u^{t-1}$, $\lambda^t$ and $k$; `g`: the stiffness gradient summed over the axes
**how?**
- a `deep` block takes `stiffness_gradient_axis` on every axis; any other takes `mirrored_gradient` on the axes along which its node is `NEAR_WALL`, so the gradient is exact at the walls too
- the cell weights $W$ and the chain rule down to the design field are applied on the host afterwards, see [sensitivity](sensitivity.md)
### adjoint_kernel
**parallelization**
same grid mapping as `gradient_kernel`
**aim**
step the adjoint field one level with the transposed operator, alone; `reconstruction_sensitivity` and `source_sensitivity` call it where `sensitivity` calls `adjoint_gradient_kernel`
**input args**
`l0`: $\lambda^{t+2}$; `l1`: $\lambda^{t+1}$; `l2`: $\lambda^{t}$, written, and allowed to be `l0`, as `fd_kernel`'s `u2` is allowed to be `u0`; `stiff`, `minv`, `derive_inertia`, and under `USE_DAMPING` `damping`, `dt`: as in `fd_kernel`; `dirichlet`: the face bitmask `boundary.face_mask(sim, Dirichlet)`, bit $2d+\textrm{side}$ set on a Dirichlet face, which sets the sign $\sigma$ of the mirror; `f0, f1, f2`: the per-axis factors of `fd_kernel`; `N0, N1, N2`, `s0, s1`: as above
**internal args**
`lc, sc, lo`: central entries of $\lambda^{t+1}$, $k$ and $\lambda^{t+2}$; `mi`: inverse inertia; `beta`: the damping coefficient of `fd_kernel`; `div_l`: $\sum_df_d$ times the transposed divergence of each axis
**how?**
- `fd_kernel`'s update with $W^{-1}A^\top W$ in place of $A$, `transposed_divergence_axis` in a `deep` block, `mirrored_divergence` along the near-wall axes elsewhere, and zero on a Dirichlet wall node
- the update itself, the inertia and the damped $1/(1+\beta)$ divisor are the forward ones: transposing the recursion in time is what marching it backwards already does
### adjoint_gradient_kernel
**parallelization**
same grid mapping as `gradient_kernel`
**aim**
step the adjoint field one level and add the gradient contribution of the level it steps from, in one pass over the grid
**input args**
`l0`: $\lambda^{t+2}$, overwritten with $\lambda^{t}$; `l1`: $\lambda^{t+1}$; `g_mass`, `g_stiff`: the two accumulators, added into; `u1`: the stored forward field paired with `l1`, the middle slot of its triplet; `stiff`, `minv`, `derive_inertia`, and under `USE_DAMPING` `damping`, `dt`: as in `fd_kernel`; `dirichlet`: as in `adjoint_kernel`; `mf`: $1/\Delta t^2$; `f0, f1, f2`: the per-axis factors of `fd_kernel`; `N0, N1, N2`, `s0, s1`: as above
**internal args**
`uc, lc, sc`: central entries of $u$, $\lambda^{t+1}$ and $k$; `div_l`, `g`: the transposed divergence of `adjoint_kernel` and the stiffness bracket of `gradient_kernel`, summed over the axes; `mi`: inverse inertia; `lo, ln`: the old and the new adjoint level at the node
**how?**
- the step is `adjoint_kernel`'s, damped or not, and the stiffness term is that of `gradient_kernel` for $\lambda^{t+1}$, whose stencil the step has already loaded; a `deep` block takes the deep forms of both, any other the wall forms along its near-wall axes
- the mass term is summed by parts in time: with $\lambda$ and $u$ both zero past their ends, the two second differences trade places exactly
$$\sum_t\lambda_i^t\left(u_i^t-2u_i^{t-1}+u_i^{t-2}\right)=\sum_t u_i^{t-1}\left(\lambda_i^{t-1}-2\lambda_i^{t}+\lambda_i^{t+1}\right)$$
and the second difference of $\lambda$ is what the step has in registers, `ln - 2 lc + lo`, so the forward field is read at the middle slot only, not as a triplet
- what the step does not see is the adjoint load the sensors add afterwards, which `adjoint_excitation_kernel` accounts for
- the same derivative as `gradient_kernel`, not an approximation, which reads three history fields and $\lambda$ again in a launch of its own; `reconstruction_sensitivity` still pairs whole triplets there
### adjoint_excitation_kernel
**parallelization**
one thread per sensor, as `excitation_kernel`; it lives in `kernels/common.cuh` next to it, as nothing in it is scalar
**aim**
inject the adjoint signal and add its share of the mass gradient, the part of $\lambda$'s second difference `adjoint_gradient_kernel` steps past
**input args**
`l2`: the adjoint level just stepped, added into; `signal`, `offset`, `lin_index`, `num_sensors`, `weight`: as in `excitation_kernel`; `g_mass`: the mass accumulator; `u1`: the stored forward field `adjoint_gradient_kernel` paired with the same step; `mf`: $1/\Delta t^2$, as in `adjoint_gradient_kernel`
**internal args**
`n`: the flat index of the sensor; `load`: the injected value
**how?**
- `l2` and `g_mass` both take `atomicAdd`, since a sensor may repeat a node
### pml_aux_kernel
**parallelization**
as `pml_kernel`, one thread per slab node of axis `d`; launched before `pml_adjoint_kernel` in every adjoint step
**aim**
advance the adjoint memories of axis `d` by one step backwards
**input args**
`l1`: $\lambda^{t+1}$; `stiff`: array of $k$; `aux`: the adjoint memories, updated in place; `profile`, `d`, `lo`, `hi`, `f`, `PML_PARAMS`: as `pml_kernel`
**internal args**
`q`: $W\lambda^{t+1}$, what the memories pair with, the stepped adjoint being that over $W$, zero on a Dirichlet wall node; `ez`: $\eta_\zeta$ before the update
**how?**
- the transpose of the forward recursion, each memory reading the divergence memory it fed
$$\eta^\pm\leftarrow b_{i\pm\frac12}\left(\eta^\pm\pm a_if_dg^\pm\eta_\zeta\right)\pm f_dg^\pm q,\qquad\eta_\zeta\leftarrow b_i\eta_\zeta+q$$
### pml_adjoint_kernel
**parallelization**
one thread per node of the slab of axis `d` grown `STENCIL_RADIUS` rows past each lined front, since a memory row is read back by every node its taps reach
**aim**
add the transposed memory terms of axis `d` to $\lambda^t$, after the adjoint step has written it
**input args**
`l2`: $\lambda^t$, added to in place; `g_mass`, `u1`, `mf`, `with_mass`: the fused step's inertia accumulator, the forward field it pairs with and its factor, patched only when `with_mass` is set; `stiff`, `minv`, `derive_inertia`: as `fd_kernel`; `aux`: the adjoint memories after `pml_aux_kernel`; `profile`, `d`, `lo`, `hi`, `f`: as `pml_kernel`; `dirichlet`: the face bitmask of `adjoint_kernel`; `PML_PARAMS`: extents and strides
**internal args**
`sum`: the gathered taps; `fold`: the reads past a wall folded onto their mirror; `delta`: the share added to $\lambda^t$
**how?**
- `sum` gathers `pml_tap` over the slab rows within `STENCIL_RADIUS`, and at rows $2..R+1$ and $N-2-R..N-3$ the reads past the wall, which the forward step took through the mirror, fold back onto their image, odd on a Dirichlet face
$$\lambda^t\mathrel{+}=\frac{1}{mW}\,\textrm{sum}$$
- `adjoint_gradient_kernel` reads $\lambda^t$ into the inertia gradient before this share arrives, so the fused path subtracts its product with $u$ here rather than reordering the launches
### superposed_kernel
**parallelization**
threads act over the entire grid, one node each, exactly as `fd_kernel` (`INTERIOR_OR_RETURN`)
**aim**
one step of the superposition variant: the `fd_kernel` update plus both Frechet integrands, in one pass
**input args**
`u0, u1, u2`: the field slots of `fd_kernel`, `u2` holding the oldest level until it is overwritten; `acc_mass`, `acc_stiff`: the two accumulators; `stiff`, `minv`, `derive_inertia`: as in `fd_kernel`; `ft`: $\pm1/(2\Delta t)^2$, zero on the first backward step; `fs`: the ratio of $\pm1/(2h_d)^2$ to $f_d$, one number since both go as $1/h_d^2$; `f0, f1, f2`, `N0, N1, N2`, `s0, s1`: as in `fd_kernel`
**internal args**
`dudt`: $u^{t-1}-u^{t-3}$ of the triplet before; `g0, g1, g2`: the central space differences of `u1`; `sum`: their weighted square sum
**how?**
- the stiffness integrand of `frechet_kernel` for this triplet, whose middle slot `u1` the step reads anyway
- the mass integrand for the triplet **before**: its newest level is final only once the excitation and the boundary kernels have run on it, which is one step later, and its oldest is what `u2` holds until the update overwrites it
- so `sensitivity.py` adds the mass integrand of each pass's last triplet with `frechet_kernel` at zero `fs`, and takes the first backward step at zero `ft`, whose triplet before is the forward's last
### frechet_kernel
**parallelization**
threads act over the entire grid, one node each, exactly as `fd_kernel` (`INTERIOR_OR_RETURN`)
**aim**
accumulate the two Frechet integrands of one field triplet, for the superposition variant
**input args**
`acc_mass`, `acc_stiff`: the two accumulators, added into; `u0, u1, u2`: one field triplet; `ft`: $\pm1/(2\Delta t)^2$; `f0, f1, f2`: $\pm1/(2h_d)^2$; `N0, N1, N2`, `s0, s1`: as above
**internal args**
`dudt`: the undivided central time difference $u^t-u^{t-2}$; `g0, g1, g2`: the undivided central space differences of `u1`; `sum`: their weighted square sum
**how?**
- summed by parts (in time for the first, in space for the second) the same two gradients turn into the **symmetric** Frechet kernels of the FWI literature
$$\frac{dJ}{dm}\bigg|_i=\sum_t\dot u_i\dot\lambda_i,\qquad\frac{dJ}{dk}\bigg|_i=-\sum_t\nabla u_i\cdot\nabla\lambda_i$$
- being symmetric *and* bilinear, they can be read off the diagonal alone, which is what removes the need to store $u$
$$B(w,w)-B(u,u)=2\alpha B(u,\lambda)+\alpha^2B(\lambda,\lambda),\qquad w=u+\alpha\lambda$$
with $\alpha$ the `scale` the caller sets. So the kernel takes **one** triplet rather than two, half the loads of a bilinear version, and the whole point of the trick
- the two derivatives are plain central differences, both centred on the middle slot
$$\dot u_i=\frac{u_i^{t}-u_i^{t-2}}{2\Delta t},\qquad\frac{\partial u_i}{\partial x_d}=\frac{u_{i+1}^{t-1}-u_{i-1}^{t-1}}{2h_d}$$
whose denominators ride in `ft` and `f0, f1, f2` together with the $\pm1$ that subtracts the forward diagonal and adds the superposed one
- no material field appears at all: $m$, $k$, the $1/2\alpha$ and the cell weights are applied on the host
- both terms are squares, hence invariant under time reversal: reversing the recursion flips $\dot u$ and leaves $\dot u^2$ alone. That is what lets the reconstructed forward field and the adjoint field share a single array
- consistent, not exact: this form agrees with `gradient_kernel` only to $O(h^2,\Delta t^2)$. [sensitivity](sensitivity.md) carries the measurements, along with `scale`, the cancellation warning and the one-step adjoint delay
