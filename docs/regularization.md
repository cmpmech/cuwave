# Regularization

**Regularization** infuses prior knowledge about the design variables `x` into the optimization to make the problem less ill-posed

`DensityFilter`, `Projection` and `SIMP` map the design variables before they enter the simulation, whereas `Tikhonov` and `TotalVariation` contribute a cost function term and its gradient

All regularization classes derive from the abstract base class `Regularization`

| member | signature | description |
|---|---|---|
| map or penalty | `__call__(x)` | adjusts the design variables, or evaluates the cost function contribution |
| derivative | `grad(x, dy=1.0)` | the respective derivative contribution, given the gradient `dy` towards the output |
| update | `set(**params)` | overwrites internal parameters, returning the instance |

## Mapping (design variable modification)
### DensityFilter
`DensityFilter` applies a conic weighting suppressing features smaller than the filter radius `rmin` following [Bruns & Tortorelli 2001](https://doi.org/10.1016/S0045-7825(00)00278-4) and [Bourdin 2001](https://doi.org/10.1002/nme.116)
$$\hat{x}_e=\frac{\sum_{i\in N_e}w_{ei}\,x_i}{\sum_{i\in N_e}w_{ei}},\qquad w_{ei}=\max\left(0,\,r_\textrm{min}-\lVert\mathbf{r}_i-\mathbf{r}_e\rVert\right)$$
with the neighborhood $N_e=\{i:\lVert\mathbf{r}_i-\mathbf{r}_e\rVert\le r_\textrm{min}\}$ of cell centers $\mathbf{r}_e$ within the filter radius

| member | signature | description |
|---|---|---|
| constructor | `DensityFilter(rmin, shape, dtype=None)` | precomputes the conic kernel $w$ and the weight sums $\sum_i w_{ei}$ on a grid of `shape`, whose length fixes the dimension of the kernel |
| filter | `__call__(x)` | returns the filtered design $\hat{x}$ |
| derivative | `grad(x, dy=1.0)` | returns $\partial\hat{x}/\partial x$ contracted with `dy`, independent of `x` |
| alternative | `sensitivity(rho, dc)` | classic OC sensitivity filter [Sigmund 2001](https://doi.org/10.1007/s001580050176) smoothing the gradient `dc` instead of the design `rho` |
| update | `set(**params)` | only `kernel` and `Hs` are stored, so a different `rmin` or `shape` needs a new instance |

The weight sums are evaluated with zero padding, so cells at the boundary are normalized by their truncated neighborhood only

The kernel is built over as many axes as `shape` has, so the same filter serves a 2D and a 3D design region. The support grows as $\left(2r_\textrm{min}\right)^\textrm{ndim}$, so a radius that is cheap on a plane is not automatically cheap on a volume

### Projection
`Projection` sharpens the filtered design towards $0/1$ with a smoothed Heaviside about the threshold `eta`, following [Wang, Lazarov & Sigmund 2011](https://doi.org/10.1007/s00158-010-0602-y) and, for the original projection idea, [Guest, Prévost & Belytschko 2004](https://doi.org/10.1002/nme.1064)
$$\bar{x}=\frac{\tanh\left(\beta\eta\right)+\tanh\left(\beta\left(x-\eta\right)\right)}{\tanh\left(\beta\eta\right)+\tanh\left(\beta\left(1-\eta\right)\right)},\qquad\frac{\partial\bar{x}}{\partial x}=\frac{\beta\left(1-\tanh^2\left(\beta\left(x-\eta\right)\right)\right)}{\tanh\left(\beta\eta\right)+\tanh\left(\beta\left(1-\eta\right)\right)}$$
with the sharpness `beta`, where $\beta\to0$ recovers the identity and $\beta\to\infty$ the Heaviside step

| member | signature | description |
|---|---|---|
| constructor | `Projection(beta, eta=0.5)` | stores sharpness `beta` and threshold `eta` |
| projection | `__call__(x)` | returns the projected design $\bar{x}$ |
| derivative | `grad(x, dy=1.0)` | returns $\partial\bar{x}/\partial x$ contracted with `dy` |
| update | `set(**params)` | continuation of the sharpness, e.g. `set(beta=2*beta)` every few iterations |

The denominator normalizes the map onto $[0,1]$, so `__call__` maps a design in $[0,1]$ back into $[0,1]$ for any `beta`

#### Continuation
`continuation(scheme, iters, start, stop, stages=4)` returns the `iters` sharpness values of a ramp from `start` to `stop`, to be handed to `set(beta=...)` once per iteration

| scheme | schedule | character |
|---|---|---|
| `"constant"` | $\beta_i=\beta_0$ | the reference, no continuation at all |
| `"linear"` | $\beta_i=\beta_0+\left(\beta_\textrm{max}-\beta_0\right)\frac{i}{n-1}$ | equal increments, so most of the run is already sharp |
| `"exponential"` | $\beta_i=\beta_0\left(\beta_\textrm{max}/\beta_0\right)^{\frac{i}{n-1}}$ | equal factors, spending the early iterations near $\beta_0$ |
| `"staircase"` | the exponential ramp held over `stages` levels | the classic continuation [Wang, Lazarov & Sigmund 2011](https://doi.org/10.1007/s00158-010-0602-y) |

with $n$ = `iters`; an unknown `scheme` raises

A ramp is needed because the two ends of the map are useless on their own: at large $\beta$ the derivative vanishes away from the threshold and the design barely moves, at small $\beta$ it moves but converges grey. The price is a cost function that changes between iterations, so the history need not decrease across a step in $\beta$ and stored curvature pairs refer to the previous map

Whether the ramp pays depends on the iteration budget: `examples/fwi/regularization/scalar2D_fwi_adam_projection.py` and `examples/fwi/regularization/scalar2D_fwi_lbfgs_projection.py` expose the choice as their `SCHEME` setting, so all four can be run against the same inversion

### SIMP
`SIMP` (solid isotropic material with penalization) raises the design to the power `p`, so that intermediate values buy less material property per unit of the volume budget and the optimizer is driven towards $0/1$, following [Bendsøe 1989](https://doi.org/10.1007/BF01650949) and [Bendsøe & Sigmund 1999](https://doi.org/10.1007/s004190050248)
$$\gamma=x_\textrm{min}+\left(1-x_\textrm{min}\right)x^p,\qquad\frac{\partial\gamma}{\partial x}=\left(1-x_\textrm{min}\right)px^{p-1}$$
with the penalization exponent `p` and the lower bound `x_min`, mapping $[0,1]$ onto $[x_\textrm{min},1]$

| member | signature | description |
|---|---|---|
| constructor | `SIMP(p=3.0, x_min=0.0)` | penalization exponent `p`, lower bound `x_min` |
| penalization | `__call__(x)` | returns the penalized design $\gamma$ |
| derivative | `grad(x, dy=1.0)` | returns $\partial\gamma/\partial x$ contracted with `dy` |
| update | `set(**params)` | continuation of the exponent, e.g. `set(p=1.0)` early and `set(p=3.0)` once the design settles |

The material coefficients are interpolated linearly in $\gamma$, so `p=1` recovers the plain interpolation and only `p>1` penalizes; unlike in compliance minimization, `x_min` is not needed to keep the system regular here, since both materials have finite properties

## Penalty (cost function modification)
### Tikhonov
`Tikhonov` adds an $L_2$ penalty on the design, or on its gradient, to stabilize an ill-posed inversion, going back to Tikhonov & Arsenin 1977 (see [Hansen 1992](https://doi.org/10.1137/1034115) on the analysis and on choosing `alpha`)
$$\textrm{order }0\textrm{ (damping):}\qquad R(x)=\frac{\alpha}{2}\lVert x-x_\textrm{ref}\rVert^2,\qquad\frac{\partial R}{\partial x}=\alpha\left(x-x_\textrm{ref}\right)$$
$$\textrm{order }1\textrm{ (smoothing):}\qquad R(x)=\frac{\alpha}{2}\sum_d\lVert D_d\left(x-x_\textrm{ref}\right)\rVert^2,\qquad\frac{\partial R}{\partial x}=\alpha\sum_dD_d^\top D_d\left(x-x_\textrm{ref}\right)$$
with the penalty weight `alpha`, the optional prior `x_ref` (zero if omitted) and $D_d$ the forward difference along axis $d$, zero-padded at the far boundary

| member | signature | description |
|---|---|---|
| constructor | `Tikhonov(alpha, order=0, x_ref=None)` | `order` selects damping (`0`) or smoothing (`1`), other values raise |
| penalty | `__call__(x)` | returns the scalar cost contribution $R(x)$ |
| derivative | `grad(x, dy=1.0)` | returns $\partial R/\partial x$ scaled by `dy`, of the shape of `x` |
| update | `set(**params)` | e.g. `set(alpha=...)` to relax the penalty as the inversion converges, note that `order` is validated in the constructor only |

Order `1` penalizes jumps between neighboring cells and hence smears interfaces, whereas order `0` merely pulls the design towards `x_ref`

### TotalVariation
`TotalVariation` penalizes the $L_1$ norm of the design gradient, which, unlike `Tikhonov` of order `1`, permits sharp interfaces while suppressing oscillations, following [Rudin, Osher & Fatemi 1992](https://doi.org/10.1016/0167-2789(92)90242-F) in the smoothed variant of [Acar & Vogel 1994](https://doi.org/10.1088/0266-5611/10/6/003)
$$R(x)=\alpha\sum_e\sqrt{\sum_d\left(D_dx\right)_e^2+\epsilon^2},\qquad\frac{\partial R}{\partial x}=\alpha\sum_dD_d^\top\left(\frac{D_dx}{\sqrt{\sum_{d'}\left(D_{d'}x\right)^2+\epsilon^2}}\right)$$
with the same $D_d$ as above and the smoothing `eps` keeping the derivative bounded where the design is locally flat

| member | signature | description |
|---|---|---|
| constructor | `TotalVariation(alpha, eps=1e-3)` | penalty weight `alpha`, smoothing `eps` |
| penalty | `__call__(x)` | returns the scalar cost contribution $R(x)$ |
| derivative | `grad(x, dy=1.0)` | returns $\partial R/\partial x$ scaled by `dy`, of the shape of `x` |
| update | `set(**params)` | e.g. `set(eps=...)` to sharpen the penalty towards the exact total variation |

Small `eps` approaches the exact total variation but stiffens the gradient, so it trades interface sharpness against the step size the optimizer tolerates
