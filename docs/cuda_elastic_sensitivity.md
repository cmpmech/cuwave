# CUDA elastic sensitivity

The backward kernels of the [elastic](elastic.md) equation. Compilation, the interior guard, the geometry macro and `RADIUS` are those of [cuda_elastic](cuda_elastic.md); `USE_DAMPING` does not appear, since the damped recursion transposes itself

## the two densities

The residual is the elastic form of the one [cuda_wave_sensitivity](cuda_wave_sensitivity.md) derives, so the gradient splits the same way: a nodal part carrying the inertia and a cell part carrying the stiffness

$$\frac{dJ}{dm}\bigg|_i=-\frac{\rho_0V}{\Delta t^2}\sum_t\boldsymbol{\lambda}_i^t\cdot\left(\mathbf{u}_i^t-2\mathbf{u}_i^{t-1}+\mathbf{u}_i^{t-2}\right),\qquad\frac{dJ}{d\gamma_c}=-\sum_t\boldsymbol{\lambda}_c^t\cdot\hat{\mathbf{K}}\,\mathbf{u}_c^t$$

with $V$ the cell volume, $\hat{\mathbf{K}}$ the cell stencil at $\gamma=1$ and $\boldsymbol{\lambda}_c$, $\mathbf{u}_c$ the two fields gathered over the corners of cell $c$. Because the assembly is exactly symmetric, the second one is a derivative of the discretization with no averaging left to differentiate

## device functions

### own_cell
**aim**
decide whether this node is the low corner of a cell inside the domain, which is where that cell's density accumulates
**input args**
`A`: grid index per axis; `NN`: logical extent per axis
**how?**
- one cell per node rather than the node's whole neighbourhood, so the cell density is accumulated once and not $2^\textrm{ndim}$ times
$$1\le a_d\le N_d-3\quad\textrm{on every axis}$$

### cell_inside
As in [cuda_elastic](cuda_elastic.md), used by the scatter rather than the assembly

### low_corner_offset
**aim**
offset of corner `l` of the cell this node is the low corner of
**input args**
`l`: corner code; `S`: stride per axis
**how?**
$$\textrm{offset}=\sum_d\textrm{bit}_d(l)\,s_d$$

## kernels

### gradient_kernel
**parallelization**
one thread per node, accumulating the nodal density everywhere and the cell density where the node owns a cell
**input args**
`u0`, `u1`, `u2`: the forward triplet, the stiffness pairing with the middle slot; `l1`: the adjoint field $\boldsymbol{\lambda}^t$; `stencil`: $\hat{\mathbf{K}}$, row-major; `mass_factor`: $\rho_0V/\Delta t^2$, the cell weights $W$ left to the epilogue; `cs`: component stride; `N0, N1, N2`, `s0, s1`: geometry, no factors
**output args**
`g_mass`: nodal inertia density, accumulated with `-=`; `g_cell`: cell stiffness density at the cell's low corner, accumulated with `-=`
**how?**
- the inertia density needs no neighbour and no material load, summing over components at the node
- the stiffness density is the cell stencil contracted between the two fields over the cell's nodes, which is the second equation above

### frechet_kernel
**parallelization**
as `gradient_kernel`
**input args**
`u0`, `u1`, `u2`: one field triplet, which for [superposition](sensitivity.md) is the combined $\mathbf{u}+k\boldsymbol{\lambda}$; `stencil`: $\hat{\mathbf{K}}$; `ft`: $\pm\rho_0V/\left(2\Delta t\right)^2$, the pass sign folded in; `fs`: $\mp1$, the pass sign and the density's own sign folded in; `cs`: component stride
**output args**
`acc_mass`, `acc_cell`: the two quadratic diagonals, accumulated with `+=`
**how?**
- both are the diagonal of the bilinear forms the gradient pairs, so their difference over the two passes gives the cross term
$$B_m\left(\mathbf{u},\mathbf{u}\right)=\rho_0V\sum_t\left|\dot{\mathbf{u}}\right|^2,\qquad B_k\left(\mathbf{u},\mathbf{u}\right)=-\sum_t\mathbf{u}_c\cdot\hat{\mathbf{K}}\,\mathbf{u}_c$$
- the sign each density needs is folded into `ft` and `fs`, so the epilogue scales both alike and no field is named there

### cell_to_node_kernel
**parallelization**
one thread per node, run once after the time loop rather than per step
**aim**
carry the cell density onto the design field through the chain rule of the harmonic cell mean
**input args**
`g_cell`: the accumulated cell density; `cell`: $\gamma_c$; `gamma`: the nodal design field; `share`: $1/2^\textrm{ndim}$; `N0, N1, N2`, `s0, s1`: geometry
**output args**
`g_stiff`: nodal stiffness gradient, accumulated with `+=`
**how?**
- the harmonic mean makes the share a design-dependent one, not a constant
$$\frac{\partial\gamma_c}{\partial\gamma_i}=\frac{1}{2^\textrm{ndim}}\left(\frac{\gamma_c}{\gamma_i}\right)^2$$
- running it once at the end rather than every step is what keeps the cell density a cell quantity for the whole march, and it is why `g_cell` is allocated alongside the two the caller sees

The cell weights $W$ apply to the inertia density only. The stiffness density lives on cells, every one of which is interior by construction, so the wall ring it would halve does not arise
