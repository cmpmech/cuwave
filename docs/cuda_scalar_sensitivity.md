# CUDA scalar sensitivity

## what is differentiated
One forward step of [cuda_scalar](cuda_scalar.md), written as a residual instead of an update
$$C^t=\frac{m}{\Delta t^2}\left(u^t-2u^{t-1}+u^{t-2}\right)-\nabla\cdot(k\nabla u^{t-1})-b^t=0$$
Making $J-\sum_t\lambda^t C^t$ stationary in $u$ defines the **adjoint field** $\lambda$: the same three-term recursion, run backwards in time, driven at the sensors by $\partial J/\partial u$ in place of the source
$$\lambda^{t}=2\lambda^{t+1}-\lambda^{t+2}+\frac{\Delta t^2}{m}\left(\nabla\cdot(k\nabla\lambda^{t+1})+\frac{\partial J}{\partial u^{t}}\right)$$
where the divergence acting on $\lambda$ is the **transpose** of the forward one under the cell weights $W$, not the forward one again: above order 2 the graded wall rows and the wide flux under a varying $k$ make the two differ. `adjoint_kernel` steps $\lambda$ with that transpose, and `adjoint_gradient_kernel` steps it and accumulates both gradients in one pass; the boundary kernels are the forward ones. What is left over is $dJ/d\theta=-\sum_t\lambda^t\,\partial C^t/\partial\theta$, one sum per material field, and those two sums are all this file computes
$$\frac{dJ}{dm}\bigg|_i=-\frac{1}{\Delta t^2}\sum_t\lambda_i^t\left(u_i^t-2u_i^{t-1}+u_i^{t-2}\right)$$
$$\frac{dJ}{dk}\bigg|_i=\sum_t\frac{\partial}{\partial k_i}\left[\lambda^t\cdot\nabla\cdot(k\nabla u^{t-1})\right]$$
Both are exact derivatives of the **discretization**, not of the PDE. And both differentiate the residual rather than the update, which is why neither carries the $\Delta t^2$ that the forward step folds into $f_d$: the mass term gets it back as `inv_dt2`, and the stiffness term runs on $F_d=f_d/\Delta t^2$, i.e. $2c_0^2/h_d^2$ (`ScalarWave`) or $2/h_d^2$ (`AcousticWave`)
The prelude of [cuda_scalar](cuda_scalar.md) is prepended here too, so `real_t`, the coefficient accessors and the graded radii are not repeated

## the transposed operator
One axis, nodes $0..N-1$ with the ghosts at both ends. The forward step's flux divergence is
$$(Au)_i=f\left(g_{i+\frac{1}{2}}D_i^+-g_{i-\frac{1}{2}}D_i^-\right),\qquad D_i^+=\sum_{k=1}^{r_i}w_{r_i,k}\left(u_{i+k}-u_{i-k+1}\right),\qquad D_i^-=\sum_{k=1}^{r_i}w_{r_i,k}\left(u_{i+k-1}-u_{i-k}\right)$$
with the graded radius $r_i$ (`CLOSURE`), the cell coefficients $w_{r,k}$ (`OP_W`) and the cell stiffness $g$. A cell's flux is formed twice, once with the radius of each node it borders, so wherever $r_i$ changes (the graded rows) or $g$ varies under a wide stencil, $A$ is not $W$-symmetric. Its transpose is a gather over the rows that read node $a$
$$\left(W^{-1}A^\top W\lambda\right)_a=\frac{f}{w_a}\sum_{k=1}^{R}\left(g_{a+k-\frac{1}{2}}E_{a+k-1,k}-g_{a-k+\frac{1}{2}}E_{a-k,k}\right),\qquad E_{i,k}=c_{i+1,k}q_{i+1}-c_{i,k}q_i$$
with the scaled field $q_i=w_i\lambda_i$, the cell weight $w_i$ ($\tfrac{1}{2}$ on the wall rows $1$ and $N-2$, else 1) and the tap $c_{i,k}=w_{r_i,k}$ for $k\le r_i$ on a stepped row, 0 otherwise. Deep in the interior $E_{i,k}=w_{R,k}\left(\lambda_{i+1}-\lambda_i\right)$, so the transpose is the **wide divergence of compact cell fluxes** where the forward step is the compact divergence of wide ones
- the forward step reads the ghost as the mirror $u_0=\sigma u_2$, $\sigma=+1$ on a Neumann face and $-1$ on a Dirichlet one, so the ghost's column of $A$ folds onto node 2: $a=2$ adds $\sigma f\sum_k g_{k-\frac{1}{2}}c_{k,k}q_k/w_2$, the rows $k$ that reach the ghost with their tap $k$, and $a=N-3$ the same at the high face
- the stiffness gradient is the same gather applied to the bracket of `stiffness_gradient_axis`
$$G_a=\frac{1}{w_a}\left[\frac{\partial g_{a+\frac{1}{2}}}{\partial k_a}\sum_k\Delta^+_{a,k}E_{a,k}+\frac{\partial g_{a-\frac{1}{2}}}{\partial k_a}\sum_k\Delta^-_{a,k}E_{a-1,k}\right],\qquad\Delta^+_{a,k}=u_{a+k}-u_{a-k+1},\quad\Delta^-_{a,k}=u_{a+k-1}-u_{a-k}$$
plus, at $a=2$ and $N-3$, the radius-1 cell between the ghost and the wall row: the stiffness ghost is always the even mirror (`wave.mirror_ghosts`), so its derivative lands on node 2 as well
- a ghost column is only ever read along its own axis, so all of this holds axis by axis and no corner term appears. Both the step and the gradient read rows $1..N-2$ of $\lambda$ only, so whatever the boundary kernels write into $\lambda$'s ghosts is never read

## split launch
The three transposed kernels (`gradient_kernel`, `adjoint_kernel`, `adjoint_gradient_kernel`) run the deep formula wherever a node's whole gather is deep and the general one elsewhere. A warp runs both when its lanes disagree, so the launch is arranged so that they never do along the fast axis
### WALL_REACH, NEAR_WALL, WALL_COLUMNS
- `WALL_REACH`: $2R$, or 3 at $R=1$: a node closer to a ghost than this gathers from a graded row, the halved wall row or a ghost, or is the node a ghost folds onto
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
### INTERIOR_OR_RETURN, AXIS_RADII, AXIS_OFFSETS
`fd_kernel`'s opening lines, as macros rather than a device function because they declare the axis indices and `return` early; `frechet_kernel` and `superposed_kernel` keep them
- `INTERIOR_OR_RETURN`: maps the thread to `a0, a1, a2` with the fastest axis on `x` (as `grid_block` does on the host), retires the ghost ring and the padding tail, and forms the flat index `idx`
- `AXIS_RADII`: `r0, r1, r2` from `CLOSURE`, for the kernel that walks a graded stencil
- `AXIS_OFFSETS`: `o0, o1, o2`, the neighbour stride per axis, for the kernel that never reaches further than one node

## gradient helpers
### stiffness_gradient_axis
**aim**
compute one axis of $\partial\left[\lambda\cdot\nabla\cdot(k\nabla u)\right]/\partial k_i$ at a deep node $i$ (`idx`)
**input args**
`u1`: array of $u$; `l1`: array of $\lambda$; `stiff`: array of $k$; `idx`: index; `s`: stride along axis; `uc`, `lc`, `sc`: `u1[idx]`, `l1[idx]`, `stiff[idx]`; `factor`: $F_d$, the same per-axis factor `flux_divergence_axis` takes but **without** the $\Delta t^2$
**internal args**
`sp, sm`: `stiff` at the two neighbours $k_{i\pm1}$; `dgp, dgm`: the two cell stiffnesses differentiated with respect to `sc`; `Dp, Dm`: the two fluxes $D_i^+, D_i^-$ at the full radius $R$
**how?**
- $\lambda\cdot\nabla\cdot(k\nabla u)$ is a sum over neighboring cells, not nodes: the cell at $i+\frac{1}{2}$ carries one stiffness $g^+$ and one flux $D_i^+$, and enters node $i$ with a $+$ and node $i+1$ with a $-$
$$\lambda\cdot\nabla\cdot(k\nabla u)=-\sum_d F_d\sum_{\textrm{cells}}g^+D_i^+\left(\lambda_{i+1}-\lambda_i\right)$$
- $k_i$ sits in exactly two of those cells, so the derivative stays local
$$\frac{\partial}{\partial k_i}\left[\lambda\cdot\nabla\cdot(k\nabla u)\right]=-\sum_d F_d\left[\frac{\partial g^+}{\partial k_i}D_i^+\left(\lambda_{i+1}-\lambda_i\right)+\frac{\partial g^-}{\partial k_i}D_i^-\left(\lambda_i-\lambda_{i-1}\right)\right]$$
- and the derivative of the (halved) harmonic cell mean is as cheap as the mean itself
$$g^\pm=\frac{k_ik_{i\pm1}}{k_i+k_{i\pm1}}\qquad\Longrightarrow\qquad\frac{\partial g^\pm}{\partial k_i}=\left(\frac{k_{i\pm1}}{k_i+k_{i\pm1}}\right)^2$$
- this is $G_a$ of the transposed operator where every tap is the full one and every weight 1. The function returns the bracket alone, so the outer minus is applied once by the caller (`g_stiff -= g`)
### transposed_divergence_axis
**aim**
compute one axis of $W^{-1}A^\top W\lambda$ at a deep node, without its factor $f_d$
**input args**
`l1`: array of $\lambda$; `stiff`: array of $k$; `idx`: index; `s`: stride along axis; `lc`, `sc`: `l1[idx]`, `stiff[idx]`
**internal args**
`sh, sl, lh, ll`: $k$ and $\lambda$ at the inner end of the two cells $k$ out, rolled outward one node per tap; `sh1, sl1, lh1, ll1`: the same at the outer end
**how?**
- the deep form of the gather, $\sum_kw_{R,k}\left(g_{a+k-\frac{1}{2}}\left(\lambda_{a+k}-\lambda_{a+k-1}\right)-g_{a-k+\frac{1}{2}}\left(\lambda_{a-k+1}-\lambda_{a-k}\right)\right)$
- it forms $2R$ cell stiffnesses where `flux_divergence_axis` forms two, and reads $k$ at all $2R+1$ taps: that, not the arithmetic, is what a transposed step costs over a forward one

## wall helpers
The general form of the gather, for the nodes `NEAR_WALL` along an axis. Every load goes through `line`, so a tap past a ghost reads the ghost again; its tap is 0, so the value only has to be finite
### row_distance, graded_tap, cell_mean, line
- `row_distance(i, N)`: $\min(i,N-1-i)$ clamped to $[0,R+1]$, so 0 on a ghost and past the end, 1 on a wall row, and anything from $R$ on a full-radius row
- `graded_tap(j, k)`: $c_{i,k}w_i$ for a row at distance `j`: 0 if $j<k$ (a ghost has none), else $w_{\min(R,j),k}$, halved on the wall row
- `cell_mean(sl, sh)`: the halved harmonic mean $g$ of a cell's two stiffnesses
- `line(field, idx, s, a, N, i)`: `field` on row $a+i$ of the axis, clamped into $[0,N-1]$
### graded_flux
**aim**
the term $g_{a+i+\frac{1}{2}}E_{a+i,k}$ of the gather: tap $k$'s rise across the cell between rows $a+i$ and $a+i+1$, times its cell stiffness
**input args**
`l1`, `stiff`, `idx`, `s`: as in `transposed_divergence_axis`; `a`: the node's index along the axis; `N`: the axis' logical extent; `i`: the cell's lower row, relative to `a`; `k`: the tap
**how?**
- four loads per call rather than a register array of the whole gather: the wall path is rare, and the arrays would raise the register count of the whole kernel, deep path included
### graded_divergence
**aim**
one axis of $W^{-1}A^\top W\lambda$ at a near-wall node, without its factor $f_d$
**input args**
as `graded_flux`, plus `faces`: the axis' two bits of `dirichlet`, low side first
**how?**
- the gather above, cell pair by cell pair; the ghost fold at $a=2$ and $N-3$ with the mirror's sign taken from `faces`; divided by the wall row's half weight at $a=1$ and $N-2$
### graded_gradient
**aim**
one axis of the stiffness gradient bracket $G_a$ at a near-wall node, scaled as `stiffness_gradient_axis`'s
**input args**
`u1`: array of $u$; `l1`, `stiff`, `idx`, `s`, `a`, `N`: as in `graded_flux`
**internal args**
`jm, jc, jp`: `row_distance` of rows $a-1, a, a+1$; `qm, qc, qp`: $\lambda$ there; `sum_p, sum_m`: $\sum_k\Delta^\pm_{a,k}E$ of the two cells the node borders
**how?**
- $G_a$ with the ghost cell's radius-1 term at $a=2$ and $N-3$, divided by the wall row's half weight. It needs no face bits: the stiffness ghost is the even mirror on every face
### flux_divergence_axis
byte-identical to the one in [cuda_scalar](cuda_scalar.md), since each `.cu` is its own compilation unit; `superposed_kernel` steps with it
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
- a `deep` block takes `stiffness_gradient_axis` on every axis; any other takes `graded_gradient` on the axes along which its node is `NEAR_WALL`, so the gradient is exact at the walls too
- the cell weights $W$ and the chain rule down to the design field are applied on the host afterwards, see [sensitivity](sensitivity.md)
### adjoint_kernel
**parallelization**
same grid mapping as `gradient_kernel`
**aim**
step the adjoint field one level with the transposed operator, alone; `reconstruction_sensitivity` and `source_sensitivity` call it where `sensitivity` calls `adjoint_gradient_kernel`
**input args**
`l0`: $\lambda^{t+2}$; `l1`: $\lambda^{t+1}$; `l2`: $\lambda^{t}$, written, and allowed to be `l0`, as `fd_kernel`'s `u2` is allowed to be `u0`; `stiff`, `minv`, `derive_inertia`, and under `USE_DAMPING` `damping`, `dt`: as in `fd_kernel`; `dirichlet`: the face bitmask `boundary.face_mask(sim, Dirichlet)`, bit $2d+\textrm{side}$ set on a Dirichlet face, which sets the sign $\sigma$ of the ghost fold; `f0, f1, f2`: the per-axis factors of `fd_kernel`; `N0, N1, N2`, `s0, s1`: as above
**internal args**
`lc, sc, lo`: central entries of $\lambda^{t+1}$, $k$ and $\lambda^{t+2}$; `mi`: inverse inertia; `beta`: the damping coefficient of `fd_kernel`; `div_l`: $\sum_df_d$ times the transposed divergence of each axis
**how?**
- `fd_kernel`'s update with $W^{-1}A^\top W$ in place of $A$, `transposed_divergence_axis` in a `deep` block, `graded_divergence` along the near-wall axes elsewhere
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
