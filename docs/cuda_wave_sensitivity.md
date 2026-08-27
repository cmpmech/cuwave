# CUDA sensitivity kernels

## what is differentiated
One forward step of [cuda_wave](cuda_wave.md), written as a residual instead of an update
$$C^t=\frac{m}{\Delta t^2}\left(u^t-2u^{t-1}+u^{t-2}\right)-\nabla\cdot(k\nabla u^{t-1})-b^t=0$$
Making $J-\sum_t\lambda^t C^t$ stationary in $u$ defines the **adjoint field** $\lambda$: the same three-term recursion, run backwards in time, driven at the sensors by $\partial J/\partial u$ in place of the source
$$\lambda^{t}=2\lambda^{t+1}-\lambda^{t+2}+\frac{\Delta t^2}{m}\left(\nabla\cdot(k\nabla\lambda^{t+1})+\frac{\partial J}{\partial u^{t}}\right)$$
so $\lambda$ needs no kernel of its own — `sensitivity.py` steps it with `fd_kernel` and the same boundary kernels. What is left over is $dJ/d\theta=-\sum_t\lambda^t\,\partial C^t/\partial\theta$, one sum per material field, and those two sums are all this file computes
$$\frac{dJ}{dm}\bigg|_i=-\frac{1}{\Delta t^2}\sum_t\lambda_i^t\left(u_i^t-2u_i^{t-1}+u_i^{t-2}\right)$$
$$\frac{dJ}{dk}\bigg|_i=\sum_t\frac{\partial}{\partial k_i}\left[\lambda^t\cdot\nabla\cdot(k\nabla u^{t-1})\right]$$
Both are exact derivatives of the **discretization**, not of the PDE. And both differentiate the residual rather than the update, which is why neither carries the $\Delta t^2$ that the forward step folds into $f_d$: the mass term gets it back as `inv_dt2`, and the stiffness term runs on $F_d=f_d/\Delta t^2$, i.e. $2c_0^2/h_d^2$ (`ScalarWave`) or $2/h_d^2$ (`AcousticWave`)
## gradient helpers

### INTERIOR_OR_RETURN, AXIS_RADII, AXIS_OFFSETS
`fd_kernel`'s opening lines, as macros rather than a device function because they declare the axis indices and `return` early
- `INTERIOR_OR_RETURN`: maps the thread to `a0, a1, a2` with the fastest axis on `x` (as `grid_block` does on the host), retires the ghost ring and the padding tail, and forms the flat index `idx`
- `AXIS_RADII`: `r0, r1, r2` from `CLOSURE`, for the kernel that walks a graded stencil
- `AXIS_OFFSETS`: `o0, o1, o2`, the neighbour stride per axis, for the kernel that never reaches further than one node
### stiffness_gradient_axis
**aim**
compute one axis of $\partial\left[\lambda\cdot\nabla\cdot(k\nabla u)\right]/\partial k_i$ at $i$ (`idx`)
**input args**
`u1`: array of $u$; `l1`: array of $\lambda$; `stiff`: array of $k$; `idx`: index; `s`: stride along axis; `uc`, `lc`, `sc`: `u1[idx]`, `l1[idx]`, `stiff[idx]`; `factor`: $F_d$, the same per-axis factor `flux_divergence_axis` takes but **without** the $\Delta t^2$; `r`: radius of finite difference scheme in axis
**internal args**
`sp, sm`: `stiff` at the two neighbours $k_{i\pm1}$; `dgp, dgm`: the two cell stiffnesses differentiated with respect to `sc`; `Dp, Dm`: the two fluxes $D_i^+, D_i^-$, formed exactly as in `flux_divergence_axis`
**how?**
- $\lambda\cdot\nabla\cdot(k\nabla u)$ is a sum over neighboring cells, not nodes: the cell at $i+\frac{1}{2}$ carries one stiffness $g^+$ and one flux $D_i^+$, and enters node $i$ with a $+$ and node $i+1$ with a $-$
$$\lambda\cdot\nabla\cdot(k\nabla u)=-\sum_d F_d\sum_{\textrm{cells}}g^+D_i^+\left(\lambda_{i+1}-\lambda_i\right)$$
- $k_i$ sits in exactly two of those cells, so the derivative stays local
$$\frac{\partial}{\partial k_i}\left[\lambda\cdot\nabla\cdot(k\nabla u)\right]=-\sum_d F_d\left[\frac{\partial g^+}{\partial k_i}D_i^+\left(\lambda_{i+1}-\lambda_i\right)+\frac{\partial g^-}{\partial k_i}D_i^-\left(\lambda_i-\lambda_{i-1}\right)\right]$$
- and the derivative of the (halved) harmonic cell mean is as cheap as the mean itself
$$g^\pm=\frac{k_ik_{i\pm1}}{k_i+k_{i\pm1}}\qquad\Longrightarrow\qquad\frac{\partial g^\pm}{\partial k_i}=\left(\frac{k_{i\pm1}}{k_i+k_{i\pm1}}\right)^2$$
- the function returns the bracket alone, so the outer minus is applied once by the caller (`g_stiff -= g`)
## kernels
### gradient_kernel
**parallelization**
threads act over entire grid (1D, 2D or 3D), one node each, exactly as `fd_kernel`
**aim**
add one time step's contribution to both gradient sums
**input args**
`g_mass`, `g_stiff`: the two accumulators over the padded grid, added into; `u0, u1, u2`: the forward triplet $u^{t-2}, u^{t-1}, u^{t}$; `l1`: $\lambda^{t}$, named after `u1` because the stiffness term pairs the two; `stiff`: array of $k$; `inv_dt2`: $1/\Delta t^2$; `F0, F1, F2`: per-axis factors $f_d/\Delta t^2$; `N0, N1, N2`, `s0, s1`: grid dimensions and strides as in `fd_kernel`
**internal args**
`uc, lc, sc`: central entries of $u^{t-1}$, $\lambda^t$ and $k$; `g`: the stiffness gradient summed over the axes
**how?**
- the cell weights $W$ and the chain rule down to the design field are applied on the host afterwards, see [sensitivity](sensitivity.md)
### frechet_kernel
**parallelization**
same grid mapping as `gradient_kernel`
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
with $\alpha$ the `scale` the caller sets. So the kernel takes **one** triplet rather than two — half the loads of a bilinear version, and the whole point of the trick
- the two derivatives are plain central differences, both centred on the middle slot
$$\dot u_i=\frac{u_i^{t}-u_i^{t-2}}{2\Delta t},\qquad\frac{\partial u_i}{\partial x_d}=\frac{u_{i+1}^{t-1}-u_{i-1}^{t-1}}{2h_d}$$
whose denominators ride in `ft` and `f0, f1, f2` together with the $\pm1$ that subtracts the forward diagonal and adds the superposed one
- no material field appears at all: $m$, $k$, the $1/2\alpha$ and the cell weights are applied on the host
- both terms are squares, hence invariant under time reversal — reversing the recursion flips $\dot u$ and leaves $\dot u^2$ alone. That is what lets the reconstructed forward field and the adjoint field share a single array
- consistent, not exact: this form agrees with `gradient_kernel` only to $O(h^2,\Delta t^2)$. [sensitivity](sensitivity.md) carries the measurements, along with `scale`, the cancellation warning and the one-step adjoint delay
