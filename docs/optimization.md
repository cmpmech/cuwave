# Optimization

**Optimization** finds the design variables that minimize (or maximize through a sign change) an objective $C$

All considered optimizers are gradient-based and hence require the sensitivities $g=\partial C/\partial x$ obtained from the sensitivity analysis

Both optimizers share the interface below and are stateful, `Adam` holding its two moment estimates and `Lbfgs` its secant pairs, so one instance belongs to one optimization run

| member | signature       | description                                                                   |
| ------ | --------------- | ----------------------------------------------------------------------------- |
| update | `step(x, grad)` | advances the internal state and returns the new design, leaving `x` untouched |

Neither optimizer enforces box constraints, so clipping the returned design to e.g. $[0,1]$ is left to the driver
## Adam
`Adam` (adaptive moment estimation) rescales the gradient by running estimates of its first and second moment, so that every design variable takes a step of comparable size irrespective of its sensitivity magnitude, following [Kingma & Ba 2015](https://doi.org/10.48550/arXiv.1412.6980)
$$m_t=\beta_1m_{t-1}+\left(1-\beta_1\right)g_t,\qquad v_t=\beta_2v_{t-1}+\left(1-\beta_2\right)g_t^2,\qquad\hat{m}_t=\frac{m_t}{1-\beta_1^t},\qquad\hat{v}_t=\frac{v_t}{1-\beta_2^t}$$
$$x_t=x_{t-1}-\eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$
with the learning rate `lr` $\eta$, the exponential decay rates `betas` $(\beta_1,\beta_2)$ and the guard `eps` $\epsilon$ against division by zero, where the moments start from zero and are therefore bias-corrected by $1-\beta_i^t$

| member | signature | description |
|---|---|---|
| constructor | `Adam(lr=1e-2, betas=(0.9, 0.999), eps=1e-8)` | stores the step size and the two decay rates, initializing both moments and the iteration counter to zero |
| update | `step(x, grad)` | advances the counter and both moments and returns the new design |

## L-BFGS
`Lbfgs` approximates the inverse Hessian from the last `k` secant pairs and applies it to the gradient through the two-loop recursion, going back to [Nocedal 1980](https://doi.org/10.1090/S0025-5718-1980-0572855-7) and [Liu & Nocedal 1989](https://doi.org/10.1007/BF01589116), in the form given by [Fichtner 2021](https://doi.org/10.33774/coe-2021-qpq2j)
$$s_i=x_{i+1}-x_i,\qquad y_i=g_{i+1}-g_i,\qquad\rho_i=\frac{1}{y_i^\top s_i}$$
$$\textrm{backward (newest to oldest):}\qquad\alpha_i=\rho_i\,s_i^\top q,\qquad q\leftarrow q-\alpha_iy_i$$
$$\textrm{scaling:}\qquad q\leftarrow\gamma_kq,\qquad\gamma_k=\frac{s_k^\top y_k}{y_k^\top y_k}$$
$$\textrm{forward (oldest to newest):}\qquad\beta_i=\rho_i\,y_i^\top r,\qquad r\leftarrow r+\left(\alpha_i-\beta_i\right)s_i$$
starting the forward pass from $r=q$ and yielding $r\approx H\,g$ with $H\approx\left(\partial^2C/\partial x^2\right)^{-1}$, hence the quasi-Newton update $x\leftarrow x-\eta\,r$ with the step size `lr` $\eta$

The scaling $\gamma_k$ of the newest pair is the initial inverse Hessian $H_0=\gamma_k\mathbf{I}$ the recursion starts from ([Nocedal & Wright 2006](https://doi.org/10.1007/978-0-387-40065-5), eq. 7.20) and it is what puts the step into the units of the design: without it the returned direction has the size of the *gradient*, whose scale is set by the physics of the problem, and the step length is left to $\eta$ -- which is then wrong as soon as the first pair enters

### Line search
`search` accepts the proposal $x_\textrm{new}$ of `step` only where it decreases the objective by the margin the Armijo (or sufficient-decrease) condition asks for ([Nocedal & Wright 2006](https://doi.org/10.1007/978-0-387-40065-5), eq. 3.4), halving the step until it does
$$p=x_\textrm{new}-x,\qquad C\left(\Pi\left(x+\alpha p\right)\right)\le C\left(x\right)+c_1\alpha\,g^\top p$$
starting from $\alpha=1$ and multiplying $\alpha$ by `shrink` on every rejection, with the constant $c_1$ `armijo` and the projection $\Pi$ `project` -- the box constraints, which the search applies to every trial and hence to the design it returns. A trial is one evaluation of the objective and no sensitivity analysis, which for a wave problem is about a third of the cost of the gradient it backtracks on

The recursion keeps $g^\top p<0$, so the margin is negative and the condition is a genuine decrease. It is nevertheless a condition that may not be met: below the floor `min_alpha` the search stops and returns the last trial, decrease or not, leaving it to the driver to notice through the returned $\alpha$

Without a pair the recursion returns the gradient itself and $\gamma_k$ never scales it, so the proposal has the size of the *gradient* and $\alpha$ has nothing to correct: that one step is bounded to `first_step` in the maximum norm instead. Since pairs of non-positive curvature are dropped, this is the state after the first step as much as before it

| member | signature | description |
|---|---|---|
| constructor | `Lbfgs(k=10, lr=1.0, first_step=0.05, armijo=1e-4, shrink=0.5, min_alpha=1e-3)` | allocates a ring buffer of `k` secant pairs, oldest first, dropping the oldest once full, and stores the four line-search parameters |
| update | `step(x, grad)` | records the pair of the preceding step, applies the recursion and returns the new design |
| line search | `search(x, grad, cost, f, project=lambda x: x)` | backtracks the proposal of `step` until the Armijo condition holds against `cost`, evaluating the objective through `f(design)`, and returns `(design, cost, alpha, trials)` |

Pairs of non-positive curvature $y^\top s\le0$ are skipped, since they would flip the sign of $\rho$ and with it the direction 
