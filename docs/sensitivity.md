# Sensitivity

> dumped from the module comments during the merge of `superposition.py` into
> `sensitivity.py` -- still raw, to be rewritten

`cuwave/sensitivity.py` holds two adjoint variants with the same signature and the
same return:

| | `sensitivity` | `superposition_sensitivity` |
| --- | --- | --- |
| forward field | stored, `N + 2` grids | reconstructed by time reversal, 3 slots |
| gradient | exact transpose of the discretisation | consistent, not exact |
| knobs | none | `scale` (the superposition `k`) |
| reports | -- | `info["cancellation"]` |

Both return `(cost, {"mass": ..., "stiff": ...}, traces, info)`, gradients as fields
over the padded grid, so the chain rule at the call site is the same
`sim.parametrization_jacobian()` pair either way. Use `sensitivity` when you want the
exact gradient and the history fits; use the other when it does not.
`tests/sensitivity_test.py` pins the first to finite differences and the second to the
first.

## cell weights W

A node on a face owns half a cell per boundary axis, and the adjoint identity is
`L^T = W L W^-1` with `W` those cell volumes -- so the exact discrete gradient carries
`W`, and the adjoint source carries `1/W`. Without them the face ring comes out
exactly 2x per boundary axis too large (4x in a corner), which a finite-difference
check catches at once.

`W` is never materialised. It is `{1, 1/2, 1/4}`, so as a field it would be one more
full grid held for the whole run -- 1 of the 8 the superposition variant lives on. The
two things it is needed for both avoid that: scaling a gradient is a slice-wise
in-place multiply (`apply_cell_weights`), and the adjoint source only ever needs `W` at
the sensor nodes (`sensor_cell_weights`). The two paths have to agree node for node; a
mismatch would misweight the adjoint source against the gradient and quietly bias the
whole result. `test_the_two_cell_weight_paths_agree` pins that.

`W` is derived for the mirrored-ghost Neumann condition, but it needs no Dirichlet
variant: there the face layer is zero in the forward field and, the adjoint recursion
being the same step, in the adjoint field too -- so whatever `W` scales on that layer
is zero either way, and the exactness above survives unchanged. Pinned by
`test_dirichlet_wall_gradient_matches_finite_differences`. The gradient *on* the
Dirichlet layer is not zero, though -- it reaches it through the flux terms -- so that
test's node on the layer is a real check and not a tautology.

## ghost nodes

A ghost node is a slaved mirror, not an unknown: it carries no equation, so a sensor
sitting on one injects the objective derivative into nothing and silently corrupts the
gradient over the whole grid -- not just nearby. `require_interior` checks it once, and
the off-by-one is easy to write (`Nx[d] - 1` is the ghost, `Nx[d] - 2` the last real
node).

## the exact variant

The forward pass is stepped straight into the history buffer, so the two leading zero
slots are `u^-2 / u^-1` and no copy is needed: `u^n` lives in `V[n + 2]`. The slot
views are made once -- `V[t]` is a host-side slice costing ~1.5 us, which is more than
the kernel it feeds, and each pass takes three or four of them per step.

The traces are read off the history in one gather rather than probed per step:
`simulate` needs a `get_signal` launch each step because it keeps only two slots, but
here every field is still on the device, and one launch saved per step is a quarter of
that pass. The sensors are interior nodes by `require_interior`, so the ghost mirroring
never touches them and reading after the loop gives the same values as reading inside
it.

In the backward pass the inertia term pairs `lambda^n` with the whole
`u^n .. u^{n-2}` triplet, the stiffness term with `u^{n-1}` alone -- the operator in
the residual is evaluated one step back, which is the middle slot of that same triplet.

`define_excitation` multiplies by `dt^2 * source_factor * minv`, which is exactly the
`m` the adjoint recursion wants, so the adjoint signal carries the remaining
`1 / (W source_factor)` -- and time is reversed.

Above order 2 the graded wall closure and the telescoped high-order flux are symmetric
only to `O(dx^2)`, so the transpose -- and with it the stiffness gradient -- stops
being exact for a varying material. The error is a discretization error, not a bug, and
it is why the sensitivity examples run at order 2.

## the superposition trick

`sensitivity` keeps the whole forward field to correlate it against the adjoint field,
which costs `N + 2` grids. The other variant trades that memory for arithmetic. Two
facts make it possible:

1. The gradient densities are *bilinear* in the forward and adjoint fields. For a
   symmetric bilinear form `B` and `w = u + k lambda`,

   ```
   B(w, w) - B(u, u) = 2 k B(u, lambda) + k^2 B(lambda, lambda)
   ```

   so `B(u, lambda)` -- the thing we want -- can be read off the *diagonal* of `B`
   alone, and the diagonal needs only one field, not two.

2. The closed lossless domain is time-reversible: the three-term recursion is symmetric
   in time, so seeding it with the last two forward states and re-injecting the source
   in reverse order walks `u` backwards exactly. The adjoint source is injected on top
   of that, so a single array carries the sum `u + k lambda`, and the forward field
   never has to be stored.

Hence: one forward pass accumulating `-B(u, u)`, one backward pass accumulating
`+B(w, w)`, and a division by `2k`. It holds five grids of its own -- three field slots
and the two accumulators, which are also the gradients it returns -- against the
`N + 2` the history costs. Measured end to end at `N = 20`, that is 7 full grids
against 27, so about 3.9x the problem fits.

Two prices, both real.

### k trades bias against round-off

The `k^2 B(lambda, lambda)` term is never removed, so the gradient carries a relative
bias growing with `k`; but the two halves of `B(w, w) - B(u, u)` are both of size
`|u|^2`, so shrinking `k` sharpens their cancellation. The right `k` is
problem-dependent over many orders of magnitude, and no amplitude heuristic finds it --
one misses by four orders, because the forward field comes from a single impulsive
source and the adjoint field from many sensors driven over all time. So `scale` is
yours to set, and `info["cancellation"]` -- the factor by which the subtraction lost
precision -- is how you set it:

**aim for a cancellation near 1e4 in float32, or 1e6 in float64.**

It scales as `1/k`, so one trial run gives you the factor to divide by. Measured on
`examples/sensitivity/scalar_sensitivity_2D.py` (240^2, N=678, float32), relative error
against the exact gradient, and the cancellation reported alongside:

| k | 1 | 1e2 | 1e4 | 1e6 |
| --- | --- | --- | --- | --- |
| rel error | 0.205 | 0.019 | 0.070 | 6.7 |
| cancellation | 1.5e6 | 1.6e4 | 1.3e2 | 3.4e-2 |

At the optimum the two variants agree to 1-2% with a cosine of 0.9998 on both ported
examples -- better than the 12% seen on a coarse 40x36 grid, as the convergence below
implies. A cancellation outside the safe band for the working precision raises a
warning rather than quietly returning round-off; `CANCELLATION_LIMIT` is about two
decimal digits short of what the precision carries (float32 ~7 digits, float64 ~16).

### it is consistent, not exact

The trick needs `B` *symmetric*, which the exact discrete transpose is not: that pairs
a wide `Dp(u)` stencil against a single difference of `lambda`. The superposition
variant uses the symmetric central-difference Frechet form instead, so it converges to
the same continuum sensitivity but is not the exact gradient of the discrete cost. With
a smooth material at fixed physical time the relative difference falls
18.5% -> 8.3% -> 2.9% -> 1.2% over res 40 -> 80 -> 160 -> 320 (cosine
0.9836 -> 0.99993).

### the one-step adjoint delay

The reconstructed forward field and the adjoint field ride the same reversed recursion
but enter it at different times: at backward step `t` the forward slot holds
`u^(N-3-t)` -- centre `N-2-t` -- while the adjoint recursion has reached
`lambda^(N-1-t)`, centre `N-t`. Advancing the adjoint source by one step
(`ADJOINT_DELAY`) lines the two centres up the way the exact transpose pairs them,
which is `lambda^t` against `u^(t-1)`, one step apart rather than none. The tail is
zero-padded, so the last step's objective derivative is dropped -- `lambda` has barely
switched on there.

This is not cosmetic. The gradient is a correlation of two oscillating fields
integrated over `N` steps, so it is acutely sensitive to their relative phase: on a
40x36 grid, getting this wrong costs a factor five in accuracy against the exact
gradient (60% vs 11% relative, cosine 0.82 vs 0.994) and the error does not shrink with
`k`, because it is an alignment error and not the `k^2` bias.

### one excitation launch, not two

The backward pass injects two sources: the adjoint source at the sensors and the
time-reversed forward source. Both ride the same excitation kernel with the same
per-node weight, so concatenating their positions and their signal columns makes them
one launch instead of two -- 9 per step rather than 10, which is what the launch-bound
regime actually pays for.

Sound for any geometry because `excitation_kernel` accumulates with `atomicAdd`: a
sensor sitting on the source is the normal FWI case, and the two would then land on one
node in one launch, where a plain read-modify-write loses an update. See the note above
that kernel in `wave.cu`, and `test_duplicate_excitation_nodes_all_land`.

## damping

Neither variant supports it, and both reject it: the adjoint of a damped step is not
that step run backwards, and time reversal needs a lossless, self-adjoint operator.
