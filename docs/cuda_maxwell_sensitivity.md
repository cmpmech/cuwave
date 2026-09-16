# CUDA maxwell sensitivity

The backward kernels of the vector curl-curl [maxwell](maxwell.md) equation: the two gradient densities the [sensitivity](sensitivity.md) variants accumulate. See [cuda_maxwell](cuda_maxwell.md) for the compilation logic and the prelude, and [cuda_elastic_sensitivity](cuda_elastic_sensitivity.md) for the shape both files share

**Without `USE_MAGNETIC` neither kernel touches a neighbour.** The stiffness is $\mathbf{C}^\top\nu\mathbf{C}$ with $\nu$ uniform, so it carries no design dependence at all and only the inertia density survives, which is one multiply per component point. That is the whole reason this pair is a third the length of its elastic counterpart

## device functions

### curl_component
Byte-identical to [cuda_maxwell](cuda_maxwell.md), and present only under `USE_MAGNETIC`. Each `.cu` is compiled as its own module from a source string, so the copy is deliberate

## kernels

### gradient_kernel
**parallelization**
one thread per grid point, writing every component point it carries and, under `USE_MAGNETIC`, every pair point too
**input args**
`u0`, `u1`, `u2`: the forward triplet at $t-1$, $t$, $t+1$; `l1`: the adjoint field $\lambda^n$; `mf`: $1/\Delta t^2$; `cs`: component stride; `f0..f2`: $1/\Delta x_d$, read only under `USE_MAGNETIC`; `N0..N2`, `s0, s1`: geometry
**output args**
`g_mass`: `NDIM` fields accumulated into, $\textrm{d}J/\textrm{d}\varepsilon$ on the component points; `g_nu`: `NPAIRS` fields of $\textrm{d}J/\textrm{d}\nu$ on the pair points, with `USE_MAGNETIC` only
**how?**
- the inertia enters the residual as $\varepsilon\ddot{\mathbf{E}}$ and nothing else, so its density is the adjoint field against the discrete second difference, with no neighbour and no material load
$$\frac{\textrm{d}J}{\textrm{d}\varepsilon_c}\mathrel{-}=\frac{1}{\Delta t^2}\lambda_c\left(u_c^{t+1}-2u_c^t+u_c^{t-1}\right)$$
- the permeability density is the forward curl contracted with the adjoint curl on the same pair point
$$\frac{\textrm{d}J}{\textrm{d}\nu_p}\mathrel{-}=b_p\left(\boldsymbol{\lambda}\right)\,b_p\left(\mathbf{u}\right)$$
- both are accumulated with `-=` into arrays the caller zeroes, so a driver may run them over a subset of the steps

### frechet_kernel
**parallelization**
as `gradient_kernel`
**input args**
`u0`, `u1`, `u2`: one field triplet, which for `superposition_sensitivity` is $\mathbf{u}$ on the forward pass and $\mathbf{u}+k\boldsymbol{\lambda}$ on the backward one; `ft`: $\pm1/\left(2\Delta t\right)^2$, the sign folded in per pass; `fs`: $\mp1$, the stiffness density entering negated so the epilogue scales both alike; `cs`, `f0..f2`, `N0..N2`, `s0, s1`: as above
**output args**
`acc_mass`, `acc_nu`: the same accumulators, holding the **diagonal** of the symmetric bilinear form rather than the cross term
**how?**
- the symmetric Frechet form of the inertia, the centred velocity squared
$$\textrm{acc}_\varepsilon\mathrel{+}=f_t\left(u_c^{t+1}-u_c^{t-1}\right)^2$$
- and of the permeability, the curl squared
$$\textrm{acc}_\nu\mathrel{+}=f_s\,b_p^2$$
- the cross term the gradient wants is read off these two diagonals by `superposition_sensitivity`, which [sensitivity](sensitivity.md) derives

Both kernels skip a component or pair point that carries no unknown, by the same `A[c] > NN[c] - 3` guard the forward pair uses, so a staggered ghost never accumulates a density that the chain rule would then scatter onto a real node
