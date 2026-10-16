# Sensitivity

**Sensitivity** differentiates a cost through the simulation, returning $\textrm{d}C/\textrm{d}m$ and $\textrm{d}C/\textrm{d}k$ as fields over the grid, yielding the gradient an [optimization](optimization.md) scheme steps on

Four variants compute it, with the same arguments (the spectral one adds its frequencies) and the same return. `sensitivity` is the default: it stores the forward field and is the exact transpose of the discretization. `reconstruction_sensitivity` marches that field backwards instead, recording only the nodes a damping layer makes irreversible, and is what to reach for once the history no longer fits an **open** domain. `superposition_sensitivity` is **the memory-free alternative** ([Herrmann, Bürchner, Kudela & Kollmannsberger 2026](https://doi.org/10.1007/s00158-025-04237-y)), reconstructing the field by time reversal with nothing recorded at all, at the price of exactness and of a closed lossless domain. `spectral_sensitivity` is **the variant for a spectral cost**: when the cost reads the record only through its DFT at a few frequencies, it keeps that DFT over the design region instead of any field history, open domain or not, and needs both of its runs to have decayed

| | `sensitivity` | `reconstruction_sensitivity` | `superposition_sensitivity` | `spectral_sensitivity` |
|---|---|---|---|---|
| forward field | stored, $N+2$ grids | marched backwards behind a strip, 3 slots | reconstructed by time reversal, 3 slots | its DFT at the cost's frequencies, over the design region |
| total footprint | $N+2$ grids and the two accumulators | 7 grids and $N+2$ copies of the strip | 5 grids, independent of $N$ | 8 grids, independent of $N$, and $4F+4$ values per region node for $F$ frequencies |
| gradient | exact transpose of the discretization | the same transpose, wherever the strip shields it | consistent, not exact | the exact derivative of the cost on the run continued to infinity |
| `sim.damping` | supported | supported, and what the strip pays for | rejected | supported |
| `sim.pml` | supported, the layer frozen | supported, the layer behind the strip | rejected | supported, the layer frozen |
| cost | any | any | any | read through the DFT at `frequencies` only |
| knobs | none | `strip_dtype` | `scale`, the superposition $k$ | `frequencies`, `region` |
| reports | none | `info["strip"]`, `info["drift"]` | `info["cancellation"]` | `info["decay"]`, `info["fit"]` |

## shared interface

| member | signature | description |
|---|---|---|
| objective | `l2_misfit(observed)` | factory returning `objective(traces) -> (cost, dcost/dtraces)` for the least-squares misfit against `observed` |
| windowed objective | `windowed_misfit(observed, window)` | the same misfit restricted to a time window, `window` zero outside the arrivals kept and carried twice so the derivative stays exact |
| exact gradient | `sensitivity(sim, source, indicator, sensors, objective)` | the cost and both gradient fields, storing the forward history, damped or lossless |
| strip gradient | `reconstruction_sensitivity(sim, source, indicator, sensors, objective, strip_dtype=None)` | the same, marching the forward field backwards behind a recorded strip rather than storing it |
| reconstructed region | `reconstruction_nodes(sim)` | the strip nodes that variant records each step, and the mask over which its gradient is exact |
| low-memory gradient | `superposition_sensitivity(sim, source, indicator, sensors, objective, scale=1.0)` | the same, in three field slots, at a `scale` the caller sets |
| spectral gradient | `spectral_sensitivity(sim, source, indicator, sensors, objective, frequencies, region=None)` | the same for a cost read off the spectrum at `frequencies`, both runs forwards in time keeping a running DFT over `region` |
| source gradient | `source_sensitivity(sim, source, indicator, sensors, objective)` | the cost and its derivative with respect to `source.signal` instead of the material, in four grids and no history |
| cell weights | `apply_cell_weights(sim, field)` | in-place multiply by the cell volume $W$ each node owns |
| cell weights | `sensor_cell_weights(sim, sensors)` | the same $W$ at the sensor nodes only, as a `(num_sensors,)` vector |
| adjoint source | `adjoint_signal(sim, dphi, sensors, scale=1.0, delay=0)` | the objective derivative reversed in time and scaled into the excitation the backward pass injects |

All four material variants return `(cost, {"mass": ..., "stiff": ...}, traces, info)`, the gradients as fields over the padded grid. `objective` takes the `(N, num_sensors)` record and returns the cost with its derivative, so **any** differentiable cost works: only the derivative reaches the adjoint

`source_sensitivity` differentiates the same cost with respect to the **emission** rather than the material, so it returns one $(N,\,\textrm{num sources})$ array in place of the pair and is the exact inverse of `adjoint_signal`: that helper divides the objective derivative by `sim.adjoint_weights` on the way into the sensors ($W\sigma$ for the pressure classes, with the source scaling `source_factor` $\sigma$), and the source gradient multiplies the adjoint field by the same weights on the way out of the source. Being linear in the signal, the cost pairs no forward field against the adjoint one there, which is what removes the history entirely; see [source inversion](source_inversion.md)

The gradients come back with respect to the two material fields, never the design. Contracting them onto the indicator is one line at the call site, with the pair `sim.parametrization_jacobian(indicator)` that [wave](wave.md) supplies

$$\frac{\textrm{d}C}{\textrm{d}\gamma}=\frac{\partial m}{\partial\gamma}\frac{\textrm{d}C}{\textrm{d}m}+\frac{\partial k}{\partial\gamma}\frac{\textrm{d}C}{\textrm{d}k}$$

with the inertia $m$, the stiffness $k$ and the design field $\gamma$. `misfit_gradient` in [utils](utils.md) does exactly this, summed over a shot list

## the adjoint field

Making $C-\sum_t\lambda^tC^t$ stationary in $u$ gives the adjoint field $\lambda$: the same three-term recursion run backwards in time, driven at the sensors by $\partial C/\partial u$ in place of the source, with the spatial operator $A$ replaced by its transpose under the cell weights
$$\lambda\textrm{ is stepped by }W^{-1}A^\top W\qquad\textrm{where }u\textrm{ is stepped by }A$$
with the cell weights $W$ of the next section. `define_adjoint_step` returns that step. The staggered [elastic](elastic.md), the [anisotropic](anisotropic.md) and the [Maxwell](maxwell.md) operators are their own transpose, so there it is `define_step`; the pressure operator is not above `space_order` 2 (the graded wall rows and the wide flux under a varying stiffness break the symmetry), so `PressureWave` launches `adjoint_kernel`, the transpose written as a gather. Boundary kernels, excitation and the two gradient sums are shared with the forward pass. Where the simulation sets `fuses_adjoint` (the pressure classes), `sensitivity` folds the step and both sums into one kernel, summing the inertia term by parts in time so that the stored forward field is read once per step instead of three times, and `superposition_sensitivity` folds its two Frechet integrands into the step the same way; the gradients are the same. [backward CUDA](cuda_scalar_sensitivity.md) derives them and documents the kernels

## cell weights

A node owns half a cell per boundary axis it sits on, and the adjoint identity $L^\top=WLW^{-1}$ puts those volumes $W$ on the gradient and $1/W$ on the adjoint source

$$W_i=\prod_d\begin{cases}\tfrac{1}{2}&i_d\textrm{ on a wall}\\1&\textrm{otherwise}\end{cases}$$

so $W$ is 1 in the interior, $\tfrac{1}{2}$ on a wall and $\tfrac{1}{4}$ in a corner. Both variants apply it before returning, so a caller never sees an unweighted gradient: without it a wall ring comes back exactly $2\times$ too large per boundary axis

$W$ is never materialized as a field. It is computed slice-wise in place for the gradient and per node for the adjoint source, which is why the two helpers above exist rather than one; a Dirichlet wall needs no variant of either, since that layer is zero in both fields and whatever $W$ scales there is zero anyway

## sensors and sources

Both are given as `(ndim, num)` **grid indices** and both are checked: a node on the ghost ring raises. A ghost node is a slaved mirror carrying no equation, so a sensor on one would inject the objective derivative into nothing and corrupt the gradient over the whole grid rather than near that node. The interior runs $1..N_d-2$, and `Nx[d] - 1` instead of `Nx[d] - 2` is an easy off-by-one to write

Receivers at arbitrary coordinates go through `Sensors` in [utils](utils.md) instead, which interpolates them onto nodes and lifts the objective for you

## reconstruction_sensitivity

Written as an update, the lossless recursion is **symmetric in its two outer slots**

$$u^{t}=2u^{t-1}-u^{t-2}+\frac{\Delta t^2}{m}\left(\nabla\cdot(k\nabla u^{t-1})+b^{t}\right)$$

so solving it for $u^{t-2}$ gives back the same expression with $u^{t}$ and $u^{t-2}$ exchanged and the source term untouched. `fd_step` called with its outer arguments swapped therefore marches the forward field *backwards* with no kernel of its own, and the excitation is re-injected at the index of the step that produced it, two ahead of the state being rebuilt. The history collapses to the three slots the triplet already needs

Damping is what breaks the symmetry: the damped update divides by $1+\beta$ and weights the trailing slot by $1-\beta$, so running it backwards multiplies by $\left(1+\beta\right)/\left(1-\beta\right)$ a step and the layer diverges. It is also the one part of the grid the application does not own, so rather than reversing it, the nodes the march would read out of it are recorded each step and replayed on the way back

$$\mathcal{L}=\mathcal{I}\setminus\mathcal{D},\qquad\textrm{strip}=\mathcal{D}\cap\textrm{dilate}\left(\mathcal{L},r\right),\qquad\textrm{valid}=\mathcal{L}$$

with the interior nodes $\mathcal{I}$, the irreversible nodes $\mathcal{D}$ (where $d>0$, and every node of a `pml`), the lossless ones $\mathcal{L}$, and the radius $r=$ `sim.reach`, which is as far as one reverse step reads: the stencil radius for the pressure classes, and $2r-1$ for the staggered [elastic](elastic.md), whose strain and divergence taps compound. The dilation runs in every direction, diagonals included, since the staggered cross terms and the cell assembly read across a corner. What is recorded is the **damped** side of the interface, $r$ nodes deep, and recording that side rather than the lossless one is what earns the whole of $\mathcal{L}$: every lossless node then reads only exact values, both in its own reverse step and in the wider stencil `gradient_kernel` takes over the middle slot. For a sponge $\mathcal{L}$ is exactly the `region` that `pad_for_sponge` in [boundary](boundary.md) hands back, so the design domain is covered node for node and the variant drops into a driver unchanged. Outside it the gradient is **zeroed**, a damped node carrying a reversed damped state that means nothing

The strip is a ring where the history is a grid, so the footprint falls by about the ratio of area to perimeter, $N_0N_1/2r\left(N_0+N_1\right)$ in 2D. That is the whole reason the variant exists: an absorbing problem too large to store as a history still fits as a strip

Damping only ever lines the faces, so a **closed lossless** domain records nothing at all: $\mathcal{D}$ is empty, so the strip is empty and `valid` is the whole interior. The reverse march then runs on the seeded states alone and reproduces `sensitivity` node for node, which `test_a_closed_lossless_domain_needs_no_strip` pins. Damping that covers *every* interior node leaves nothing to rebuild from, and `require_reconstructable` raises rather than returning a field of zeros

The march is exact in exact arithmetic, so what is left is round-off, and it accumulates over the reverse steps rather than cancelling. `info["drift"]` reports it as the norm of the reconstructed initial state (which is zero exactly) against the last forward state the march set out from, so a value near the precision is the expected reading, and the recourse if it grows is `float64` or `sensitivity`

The strip is the one record that grows with $N$, and in 3D, where it is a shell rather than a ring, it outweighs the grids. `strip_dtype` stores it narrower than the simulation, as fdtdx stores its boundary record ([Mahlau, Schubert, Berg & Rosenhahn 2026](https://doi.org/10.21105/joss.08912)): `"float32"` under a `float64` simulation, or `"bfloat16"` under either, which `march` stages back to the kernels' precision chunk by chunk. That halves the strip and gives up the exact transpose: a `float32` strip under `float64` stays near round-off, a `bfloat16` one keeps about three digits of the gradient, and `info["drift"]` grows to show it. Leave it unset unless the strip is what no longer fits

## superposition_sensitivity

The gradient densities are *bilinear* in the forward and adjoint fields, and a closed lossless domain is time-reversible. Together those remove the history: for a symmetric bilinear form $B$ and $w=u+k\lambda$

$$B(w,w)-B(u,u)=2k\,B(u,\lambda)+k^2B(\lambda,\lambda)$$

so the cross term $B(u,\lambda)$ (the gradient) is read off the **diagonal** of $B$ alone, and a diagonal needs one field where a cross term needs two. Seeding the recursion with the last two forward states and re-injecting the source in reverse walks $u$ backwards, while the adjoint source is superposed on top, so a single array carries $u+k\lambda$

One forward pass accumulates $-B(u,u)$, one backward pass accumulates $+B(w,w)$, and the difference is divided by $2k$. Five grids against $N+2$, and the footprint no longer grows with the number of steps

### choosing scale

`scale` is $k$, and it trades bias against round-off in opposite directions: the $k^2B(\lambda,\lambda)$ term is never removed, so a large $k$ biases the gradient, while a small $k$ sharpens the cancellation between two quantities both of size $\lVert u\rVert^2$. No amplitude heuristic finds the balance: the forward field comes from one impulsive source and the adjoint field from many receivers driven over all time, so their scales are unrelated

Set it from the diagnostic instead. `info["cancellation"]` reports the factor by which the subtraction lost precision, and it scales as $1/k$, so one trial run gives the factor to divide by:

> **aim for a cancellation near 1e4 in float32, or 1e6 in float64**

Past `CANCELLATION_LIMIT`, about two decimal digits short of what the precision carries, the call raises a `RuntimeWarning` naming the factor to raise `scale` by, rather than quietly returning round-off

## spectral_sensitivity

A cost that reads the record only through its DFT at a few frequencies, `intensity` in [utils](utils.md), needs the fields at those frequencies and nothing else, following the hybrid time/frequency adjoint of Meep ([Hammond, Oskooi, Johnson & Ralph 2022](https://doi.org/10.1364/OE.442074)) as fdtdx adapts it ([Mahlau, Schubert, Berg & Rosenhahn 2026](https://doi.org/10.21105/joss.08912)). Transformed in time, the three-term recursion becomes one linear system per frequency

$$A(z)\,\hat u=\hat b,\qquad A(z)=W\left(m\frac{1-2z^{-1}+z^{-2}}{\Delta t^2}+d\,\frac{1-z^{-2}}{2\Delta t}\right)+z^{-1}K,\qquad z=e^{i\omega\Delta t}$$

with the DTFT $\hat u$ of the field, the transformed source $\hat b$, the cell weights $W$ and the stiffness operator $K$. A cost whose derivative is $\textrm{Re}\left(r\,\textrm{d}\hat u\right)$ at the sensors then has the gradient

$$\frac{\textrm{d}C}{\textrm{d}\kappa}=-\textrm{Re}\left(\hat\lambda^\top\frac{\partial A(z)}{\partial\kappa}\hat u\right),\qquad A(z)^\top\hat\lambda=r$$

with a material field $\kappa$ (`mass` or `stiff`). $\hat\lambda$ is the DTFT of an ordinary **causal** run of the adjoint step, driven by any pulse whose DTFT at $\omega$ is $r$, so both passes run forwards in time and keep only a running DFT (`define_dft` in [wave](wave.md)). `spectral_pulse` picks a Gaussian a period wide; the shape only decides how soon the adjoint decays. The gradient pairs $\lambda^n$ with the triplet $(u^{n-2},u^{n-1},u^n)$, so `gradient_step` is called on the triplet $(z^{-2}\hat u,z^{-1}\hat u,\hat u)$, once with the real parts and once with the imaginary parts and $-\textrm{Im}\,\hat\lambda$: the same kernels, never a time loop of their own

`objective` is the usual record-in, derivative-out callable. Its derivative is fitted by sinusoids at `frequencies` in double precision, which recovers $r$ exactly for a spectral cost; a residual past `FIT_LIMIT` (reported as `info["fit"]`) means the cost reads the record otherwise, and the call raises. `region` limits the DFT to the design nodes, widened by `sim.reach` plus one so the gradient stencil reads transformed values, ghost nodes included, and is what makes the transform nearly free; the gradient comes back zero outside it. One adjoint run is made per frequency

The identity holds for the DTFT, the sum over all time, while the run stops at $N$. A field that settles to a constant (the static field a closed domain keeps after a source's net impulse) is continued in closed form, $\hat u\mathrel{+}=u^{N-1}z^{-N}/\left(1-z^{-1}\right)$ by `settled`; what is left is the oscillating tail. `info["decay"]` reports it as the share of the spectrum over the region the last period of either run still added, which is the relative error of the gradient to leading order. Past `DECAY_LIMIT` (1e-2) the call warns: lengthen the run or damp more. The time-domain gradient of the truncated cost carries the same tail, so neither is the better reading of a cost that has not settled; they meet as $N$ grows

The price is that condition. A resonant design rings longer than its starting guess, so a run that settles at the first iteration may not at the last, and the warning is the signal to give it more steps. In exchange there is no history, no strip, no reverse march to drift and no `scale` to set, and damping is free: this is the variant to reach for when the cost is spectral and the domain is open

## limits

The trick needs $B$ **symmetric**, which the exact discrete transpose is not: that pairs a wide flux stencil against a single difference of $\lambda$. `superposition_sensitivity` uses the symmetric central-difference Frechet form instead, so it converges to the same continuum sensitivity under refinement but is not the exact gradient of the discrete cost. For a line search or a finite-difference check, use `sensitivity`

`sensitivity` and `reconstruction_sensitivity` are the exact gradient of the discrete cost at every `space_order`, for every equation, on an unmasked grid. A `domain` or `dirichlet` mask grades each cell per node, which the transposed pressure kernels do not read: the material gradients refuse a mask, and `source_sensitivity` on one steps $\lambda$ with the forward kernel, which is the transpose at order 2 only. The price of the transpose is its wide divergence, which reads the stiffness at all $2R+1$ taps per axis where the forward step reads three, so a pressure backward step costs somewhat more than a forward one

`sensitivity` and `reconstruction_sensitivity` take a damped simulation; `superposition_sensitivity` refuses one. The field rides on `Simulation` rather than on either signature, so a driver states it once and `measure`, `misfit` and `misfit_gradient` in [utils](utils.md) need no argument for it. Transposing the damped recursion in time swaps its two neighbour coefficients, which is exactly what marching the adjoint backwards already does, so the transposed step keeps the forward step's $1/(1+\beta)$ update unchanged. Written as a residual, $d$ stands alone, so neither gradient density sees it

$$R^n=m\left(u^{n+1}-2u^n+u^{n-1}\right)+\tfrac{1}{2}d\,\Delta t\left(u^{n+1}-u^{n-1}\right)-\Delta t^2\nabla\cdot\left(k\nabla u^n\right)$$

The reverse-time reconstruction asks for more, and `superposition_sensitivity` raises rather than returning something plausible: it rebuilds the forward field by running it backwards, which only a lossless operator allows. `reconstruction_sensitivity` reads the same requirement as a statement about *where* rather than *whether*: it reverses the lossless part of the grid and replays the rest

A `pml` ([boundary](boundary.md)) is the same case with memory: its recursion is not reversible, so `superposition_sensitivity` refuses it and `reconstruction_sensitivity` counts its nodes as irreversible. With it on, every equation steps $\lambda$ through the transpose of the layer's memory terms, the adjoint memories running backwards with the same $b$, so the adjoint is no longer the forward step even where the interior is its own transpose. The memory terms carry a material dependence no gradient kernel differentiates, so the material gradient is **zeroed** on `pml_nodes` and the layer has to be design-frozen, as a driver freezes a sponge already; outside it the gradient stays exact. `spectral_sensitivity` restarts the adjoint memories for each frequency's run

The one place damping does **not** cancel is the excitation. The update divides through by $1+\beta$ and the source is added after that divide, so `excitation_weights` in [wave](wave.md) carries the same divisor

$$\beta=\frac{d\,\Delta t}{2m},\qquad\textrm{source weight}=\frac{\Delta t^2}{m\left(1+\beta\right)}$$

Without it the injected impulse depends on $d$ through $\beta$, which leaves the mass gradient a term short at the source nodes and mis-scales the adjoint source at the sensors. A mis-scaled adjoint source corrupts $\lambda$ over the whole grid rather than near it, so the gradient comes back wrong everywhere, low by exactly $\beta$

Opening the domain therefore costs `superposition_sensitivity` grid rather than a layer: the sponge in [boundary](boundary.md) removes the energy it absorbs, so that variant is left padding the domain until the wall echo lands outside the record. The other two take the sponge and open the same faces in a couple of wavelengths: `sensitivity` while the history still fits, `reconstruction_sensitivity` once it does not
