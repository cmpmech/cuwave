# CUDA elastic sensitivity

The backward kernels of the staggered [elastic](elastic.md) equation. Compilation, the geometry macros, the graded radii and the strain helpers are those of [cuda_elastic](cuda_elastic.md); `USE_DAMPING` does not appear, since the damped recursion transposes itself

## the densities

The residual is the elastic form of the one [cuda_scalar_sensitivity](cuda_scalar_sensitivity.md) derives, so the gradient splits the same way: an inertia part on each component's points and a stiffness part on each stress-point family

$$\frac{dJ}{dm}\bigg|_s=-\frac{1}{\Delta t^2}\sum_t\lambda_s^t\left(u_s^t-2u_s^{t-1}+u_s^{t-2}\right),\qquad\frac{dJ}{d\gamma_s}=-w_s\sum_t\boldsymbol{\varepsilon}\left(\boldsymbol{\lambda}^t\right):\mathbf{C}:\boldsymbol{\varepsilon}\left(\mathbf{u}^t\right)\bigg|_s$$

with $s$ a component point for the inertia and a stress point for the stiffness. Because both kernels tap the same matrix, the second one is a derivative of the discretization with no averaging left to differentiate; the kernels accumulate the raw densities and `finalize_gradients` chains them onto the nodes on the host, through the arithmetic mean of the inertia ($1/2$ per neighbour node), the identity of the node family, and the harmonic mean of a shear point

$$\frac{\partial\gamma_s}{\partial\gamma_i}=\frac{1}{4}\left(\frac{\gamma_s}{\gamma_i}\right)^2$$

## device functions

`rad_node`, `rad_half`, `clamped_face`, `normal_strains`, `condensed_lame` and `shear_strain` as in [cuda_elastic](cuda_elastic.md), evaluated here on the forward and the adjoint field alike, plus

### normal_form
**aim**
the normal bilinear form of two strain sets under the condensed coupling
**input args**
`ea`, `eb`: two strain sets from `normal_strains`; `lam`, `mu`: the Lamé scalars; `zeroed`: the bitmask both sets share
**how?**
$$2\mu\sum_d\varepsilon^a_{dd}\varepsilon^b_{dd}+\lambda_\textrm{eff}\,\textrm{tr}\,\boldsymbol{\varepsilon}^a\,\textrm{tr}\,\boldsymbol{\varepsilon}^b$$

## kernels

### gradient_kernel
**parallelization**
one thread per grid point, accumulating the inertia density on its component points and the stiffness density on its stress points
**input args**
`u0`, `u1`, `u2`: the forward triplet, the stiffness pairing with the middle slot; `l1`: the adjoint field $\boldsymbol{\lambda}^t$; `lam`, `mu`: the Lamé scalars; `mf`: $1/\Delta t^2$, the $\rho_0W/2$ of the inertia chain rule left to the epilogue; `clamped`, `cs`: as forward; `f0..f2`: $1/\Delta x_d$
**output args**
`g_mass`: `NDIM` fields of inertia density, accumulated with `-=`; `g_normal`: the node-family density; `g_shear`: `NPAIRS` fields, `NDIM >= 2` only
**how?**
- the inertia density needs no neighbour and no material load, one entry per component point
- the stiffness densities contract the adjoint strains against the forward ones: `normal_form` on the node, $\mu$ times the two shear strains on each pair point, neither carrying $\gamma$ or $w_s$, which the epilogue holds

### frechet_kernel
**parallelization**
as `gradient_kernel`
**input args**
`u0`, `u1`, `u2`: one field triplet, which for [superposition](sensitivity.md) is the combined $\mathbf{u}+k\boldsymbol{\lambda}$; `lam`, `mu`: as above; `ft`: $\pm1/\left(2\Delta t\right)^2$, the pass sign folded in; `fs`: $\mp1$, the pass sign and the density's own sign folded in
**output args**
`acc_mass`, `acc_normal`, `acc_shear`: the quadratic diagonals, accumulated with `+=`
**how?**
- both are the diagonal of the bilinear forms the gradient pairs, so their difference over the two passes gives the cross term
$$B_m\left(u,u\right)=\sum_t\left(u^{t}-u^{t-2}\right)^2,\qquad B_k\left(\mathbf{u},\mathbf{u}\right)=-\sum_t\boldsymbol{\varepsilon}:\mathbf{C}:\boldsymbol{\varepsilon}$$
- the sign each density needs is folded into `ft` and `fs`, so the epilogue scales both alike and no field is named there

There is no `cell_to_node_kernel` here: the chain rule onto the nodes is a handful of shifted adds run once after the time loop, so `finalize_gradients` does it with array slices on the host rather than a kernel of its own
