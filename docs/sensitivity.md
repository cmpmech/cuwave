# Sensitivity

**Sensitivity** differentiates a cost through the simulation, returning $\textrm{d}C/\textrm{d}m$ and $\textrm{d}C/\textrm{d}k$ as fields over the grid, yielding the gradient an [optimization](optimization.md) scheme steps on

Two variants compute it, with the same arguments and the same return. `sensitivity` is the default: it stores the forward field and is the exact transpose of the discretization. `superposition_sensitivity` is **the memory-efficient alternative** ([Herrmann, Bürchner, Kudela & Kollmannsberger 2026](https://doi.org/10.1007/s00158-025-04237-y)), reconstructing that field by time reversal instead of storing it, and is what to reach for once the history no longer fits

| | `sensitivity` | `superposition_sensitivity` |
|---|---|---|
| forward field | stored, $N+2$ grids | reconstructed by time reversal, 3 slots |
| total footprint | $N+2$ grids and the two accumulators | 5 grids, independent of $N$ |
| gradient | exact transpose of the discretization | consistent, not exact |
| knobs | none | `scale`, the superposition $k$ |
| reports | — | `info["cancellation"]` |

## shared interface

| member | signature | description |
|---|---|---|
| objective | `l2_misfit(observed)` | factory returning `objective(traces) -> (cost, dcost/dtraces)` for the least-squares misfit against `observed` |
| exact gradient | `sensitivity(sim, source, indicator, sensors, objective, damping=None)` | the cost and both gradient fields, storing the forward history |
| low-memory gradient | `superposition_sensitivity(sim, source, indicator, sensors, objective, scale=1.0, damping=None)` | the same, in three field slots, at a `scale` the caller sets |
| cell weights | `apply_cell_weights(sim, field)` | in-place multiply by the cell volume $W$ each node owns |
| cell weights | `sensor_cell_weights(sim, sensors)` | the same $W$ at the sensor nodes only, as a `(num_sensors,)` vector |
| adjoint source | `adjoint_signal(sim, dphi, sensors, scale=1.0, delay=0)` | the objective derivative reversed in time and scaled into the excitation the backward pass injects |

Both return `(cost, {"mass": ..., "stiff": ...}, traces, info)`, the gradients as fields over the padded grid. `objective` takes the `(N, num_sensors)` record and returns the cost with its derivative, so **any** differentiable cost works — only the derivative reaches the adjoint

The gradients come back with respect to the two material fields, never the design. Contracting them onto the indicator is one line at the call site, with the pair `sim.parametrization_jacobian()` that [wave](wave.md) supplies

$$\frac{\textrm{d}C}{\textrm{d}\gamma}=\frac{\partial m}{\partial\gamma}\frac{\textrm{d}C}{\textrm{d}m}+\frac{\partial k}{\partial\gamma}\frac{\textrm{d}C}{\textrm{d}k}$$

with the inertia $m$, the stiffness $k$ and the design field $\gamma$ — `misfit_gradient` in [utils](utils.md) does exactly this, summed over a shot list

## the adjoint field

Making $C-\sum_t\lambda^tC^t$ stationary in $u$ gives the adjoint field $\lambda$: the same three-term recursion run backwards in time, driven at the sensors by $\partial C/\partial u$ in place of the source. So $\lambda$ needs no kernel of its own — both variants step it with `fd_kernel` and the same boundary kernels, and only the two gradient sums are new. [backward CUDA](cuda_wave_sensitivity.md) derives them and documents the kernels

## cell weights

A node owns half a cell per boundary axis it sits on, and the adjoint identity $L^\top=WLW^{-1}$ puts those volumes $W$ on the gradient and $1/W$ on the adjoint source

$$W_i=\prod_d\begin{cases}\tfrac{1}{2}&i_d\textrm{ on a wall}\\1&\textrm{otherwise}\end{cases}$$

so $W$ is 1 in the interior, $\tfrac{1}{2}$ on a wall and $\tfrac{1}{4}$ in a corner. Both variants apply it before returning, so a caller never sees an unweighted gradient — without it a wall ring comes back exactly $2\times$ too large per boundary axis

$W$ is never materialized as a field. It is computed slice-wise in place for the gradient and per node for the adjoint source, which is why the two helpers above exist rather than one; a Dirichlet wall needs no variant of either, since that layer is zero in both fields and whatever $W$ scales there is zero anyway

## sensors and sources

Both are given as `(ndim, num)` **grid indices** and both are checked: a node on the ghost ring raises. A ghost node is a slaved mirror carrying no equation, so a sensor on one would inject the objective derivative into nothing and corrupt the gradient over the whole grid rather than near that node — the interior runs $1..N_d-2$, and `Nx[d] - 1` instead of `Nx[d] - 2` is an easy off-by-one to write

Receivers at arbitrary coordinates go through `Sensors` in [utils](utils.md) instead, which interpolates them onto nodes and lifts the objective for you

## superposition_sensitivity

The gradient densities are *bilinear* in the forward and adjoint fields, and a closed lossless domain is time-reversible. Together those remove the history: for a symmetric bilinear form $B$ and $w=u+k\lambda$

$$B(w,w)-B(u,u)=2k\,B(u,\lambda)+k^2B(\lambda,\lambda)$$

so the cross term $B(u,\lambda)$ — the gradient — is read off the **diagonal** of $B$ alone, and a diagonal needs one field where a cross term needs two. Seeding the recursion with the last two forward states and re-injecting the source in reverse walks $u$ backwards, while the adjoint source is superposed on top, so a single array carries $u+k\lambda$

One forward pass accumulates $-B(u,u)$, one backward pass accumulates $+B(w,w)$, and the difference is divided by $2k$. Five grids against $N+2$, and the footprint no longer grows with the number of steps

### choosing scale

`scale` is $k$, and it trades bias against round-off in opposite directions: the $k^2B(\lambda,\lambda)$ term is never removed, so a large $k$ biases the gradient, while a small $k$ sharpens the cancellation between two quantities both of size $\lVert u\rVert^2$. No amplitude heuristic finds the balance: the forward field comes from one impulsive source and the adjoint field from many receivers driven over all time, so their scales are unrelated

Set it from the diagnostic instead. `info["cancellation"]` reports the factor by which the subtraction lost precision, and it scales as $1/k$, so one trial run gives the factor to divide by:

> **aim for a cancellation near 1e4 in float32, or 1e6 in float64**

Past `CANCELLATION_LIMIT`, about two decimal digits short of what the precision carries, the call raises a `RuntimeWarning` naming the factor to raise `scale` by, rather than quietly returning round-off

## limits

The trick needs $B$ **symmetric**, which the exact discrete transpose is not: that pairs a wide flux stencil against a single difference of $\lambda$. `superposition_sensitivity` uses the symmetric central-difference Frechet form instead, so it converges to the same continuum sensitivity under refinement but is not the exact gradient of the discrete cost. For a line search or a finite-difference check, use `sensitivity`

Above `space_order` 2 the graded wall closure and the telescoped high-order flux are symmetric only to $O(h^2)$, so even `sensitivity` stops being the exact transpose for a varying material. This is a discretization error rather than a bug, and it is why the sensitivity drivers run at order 2

Neither variant supports damping, and both reject it rather than returning something plausible: the adjoint of a damped step is not that step run backwards, and time reversal needs a lossless, self-adjoint operator

Opening the domain therefore has to absorb without dissipating, which is what the random layer in [boundary](boundary.md) is for: it scatters the outgoing wave into an incoherent coda instead of removing its energy, so both variants keep working unchanged. Its `correlation` has to sit above one node for `superposition_sensitivity` — a material redrawn at every node has no continuum limit, so refinement sharpens the roughness along with the grid and the symmetric Frechet form never becomes consistent with the exact adjoint. Matching the grain to the wavelength, which is what makes the layer scatter in the first place, satisfies that anyway
