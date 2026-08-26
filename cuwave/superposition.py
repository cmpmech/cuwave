"""Adjoint sensitivities by the superposition trick: three field slots, no history.

`sensitivity.py` keeps the whole forward field to correlate it against the adjoint
field, which costs (N + 2) grids. This module trades that memory for arithmetic.
Two facts make it possible:

1. The gradient densities are *bilinear* in the forward and adjoint fields. For a
   symmetric bilinear form B and w = u + k lambda,

       B(w, w) - B(u, u) = 2 k B(u, lambda) + k^2 B(lambda, lambda)

   so B(u, lambda) -- the thing we want -- can be read off the *diagonal* of B
   alone, and the diagonal needs only one field, not two.

2. The closed lossless domain is time-reversible: the three-term recursion is
   symmetric in time, so seeding it with the last two forward states and
   re-injecting the source in reverse order walks u backwards exactly. The
   adjoint source is injected on top of that, so a single array carries the sum
   u + k lambda, and the forward field never has to be stored.

Hence: one forward pass accumulating -B(u, u), one backward pass accumulating
+B(w, w), and a division by 2k. The class holds five grids of its own -- three
field slots and the two accumulators, which are also the gradients it returns --
against the N + 2 the history costs. Measured end to end at N = 20, that is 7
full grids against 27, so about 3.9x the problem fits.

Two prices, both real:

* **k trades bias against round-off.** The k^2 B(lambda, lambda) term is never
  removed, so the gradient carries a relative bias growing with k; but the two
  halves of B(w, w) - B(u, u) are both of size |u|^2, so shrinking k sharpens
  their cancellation. The right k is problem-dependent over many orders of
  magnitude, and no amplitude heuristic finds it -- one misses by four orders,
  because the forward field comes from a single impulsive source and the adjoint
  field from many sensors driven over all time. So `scale` is yours to set, and
  `last_cancellation` -- the factor by which the subtraction lost precision -- is
  how you set it:

      **aim for last_cancellation near 1e4 in float32, or 1e6 in float64.**

  It scales as 1/k, so one trial run gives you the factor to divide by. Measured
  on examples/scalar_sensitivity_2D.py (240^2, N=678, float32), relative error
  against the exact gradient, and the cancellation reported alongside:

      k             1       1e2      1e4      1e6
      rel error     0.205   0.019    0.070    6.7
      cancellation  1.5e6   1.6e4    1.3e2    3.4e-2

  At the optimum the two methods agree to 1-2% with a cosine of 0.9998 on both
  ported examples -- better than the 12% seen on a coarse 40x36 grid, as the
  convergence below implies. A cancellation outside the safe band for the working
  precision raises a warning rather than quietly returning round-off.

* **It is consistent, not exact.** The trick needs B *symmetric*, which the exact
  discrete transpose of `sensitivity.py` is not: that pairs a wide Dp(u) stencil
  against a single difference of lambda. This module uses the symmetric
  central-difference Frechet form instead, so it converges to the same continuum
  sensitivity but is not the exact gradient of the discrete cost. With a smooth
  material at fixed physical time the relative difference falls 18.5% -> 8.3% ->
  2.9% -> 1.2% over res 40 -> 80 -> 160 -> 320 (cosine 0.9836 -> 0.99993).

So: use `sensitivity.sensitivity` when you want an exact gradient and the history
fits; use this when it does not. tests/superposition_test.py pins them together.
"""

import warnings

import cupy as cp

from .boundary import define_boundary
from .sensitivity import (
    KERNEL_PATH,
    apply_cell_weights,
    require_interior,
    sensor_cell_weights,
)
from .wave import (
    compile_kernels,
    define_excitation,
    define_get_signal,
    define_step_method,
    grid_block,
)

# The reconstructed forward field and the adjoint field ride the same reversed
# recursion but enter it at different times: at backward step t the forward slot
# holds u^(N-3-t) -- centre N-2-t -- while the adjoint recursion has reached
# lambda^(N-1-t), centre N-t. Advancing the adjoint source by one step lines the
# two centres up the way the exact transpose in sensitivity.py pairs them, which
# is lambda^t against u^(t-1), one step apart rather than none.
#
# This is not cosmetic. The gradient is a correlation of two oscillating fields
# integrated over N steps, so it is acutely sensitive to their relative phase: on
# a 40x36 grid, getting this wrong costs a factor five in accuracy against the
# exact gradient (60% vs 11% relative, cosine 0.82 vs 0.994) and the error does
# not shrink with k, because it is an alignment error and not the k^2 bias.
ADJOINT_DELAY = 1

# Cancellation past which the subtraction has eaten too much of the mantissa to
# trust: float32 carries ~7 decimal digits and float64 ~16, and the measurements in
# the module docstring put the usable limit about two digits short of each.
CANCELLATION_LIMIT = {"float32": 1e5, "float64": 1e12}


class SuperpositionSensitivity:
    """Callable gradient evaluator. Built once per (sim, source, sensors), then
    called per design:

        grad = SuperpositionSensitivity(sim, source, sensors)
        cost, grads, traces = grad(indicator, objective)

    `objective(traces) -> (cost, dcost/dtraces)` as in `sensitivity.py`, and the
    returned `{"mass": ..., "stiff": ...}` follow the same convention, so the two
    modules are drop-in comparable and the chain rule at the call site is the same
    `sim.parametrization_jacobian()` pair.
    """

    def __init__(self, sim, source, sensors, scale=1.0, damping=None):
        if damping is not None:
            raise NotImplementedError(
                "the superposition trick reconstructs the forward field by time "
                "reversal, which needs a lossless operator; damping destroys it"
            )
        require_interior(sim, sensors, "sensor")
        require_interior(sim, source.position, "source")

        self.sim = sim
        self.source = source
        self.sensors = sensors
        self.scale = scale

        # everything that does not depend on the design is hoisted out of __call__
        self.kernels = compile_kernels(sim)
        self.sens_kernels = compile_kernels(sim, KERNEL_PATH)
        self.bc_step = define_boundary(sim, self.kernels)
        self.get_signal = define_get_signal(sim, sensors, self.kernels)
        self.w_sensors = sensor_cell_weights(sim, sensors)

        # The backward pass injects two sources: the adjoint source at the sensors
        # and the time-reversed forward source. Both ride the same excitation kernel
        # with the same per-node weight, so concatenating their positions and their
        # signal columns makes them one launch instead of two -- 9 per step rather
        # than 10, which is what the launch-bound regime actually pays for.
        #
        # Sound for any geometry because excitation_kernel accumulates with atomicAdd:
        # a sensor sitting on the source is the normal FWI case, and the two would
        # then land on one node in one launch, where a plain read-modify-write loses
        # an update. See the note above that kernel in wave.cu.
        self.backward_position = cp.concatenate((sensors, source.position), axis=1)

        self.grid, self.block = grid_block(sim)
        self.frechet_kernel = self.sens_kernels.get_function("frechet_kernel")
        # (2 dt)^-2 and (2 dx_k)^-2, folded in so the kernel stays arithmetic-free,
        # each carrying the sign that subtracts the forward diagonal and adds the
        # superposed one. Both argument lists are built here and mutated in place --
        # see the launch-cost note above define_step_method in wave.py, and note the
        # sign is a compile-time constant of the pass, not something to recompute
        # 2 N times.
        self._args = {}
        for sign in (-1.0, 1.0):
            geom = [sim.dtype(sign / (2.0 * sim.dt) ** 2)]
            for d in range(sim.ndim):
                geom.append(sim.dtype(sign / (2.0 * sim.dx[d]) ** 2))
                geom.append(sim.Nx[d])
                if d:
                    geom.append(sim.strides[d - 1])
            # geom is [ft, f0, N0, f1, N1, s0, ...]: the axis triples follow the
            # (factor, N, stride) order of wave.axis_geometry, with s0 the stride of
            # the *previous* axis, so the first axis contributes only two entries
            self._args[sign] = [None] * 5 + geom

    def _accumulate(self, acc_mass, acc_stiff, u0, u1, u2, sign):
        args = self._args[sign]
        args[0], args[1] = acc_mass, acc_stiff
        args[2], args[3], args[4] = u0, u1, u2
        self.frechet_kernel(self.grid, self.block, args)

    def _adjoint_signal(self, dphi, k):
        # reversed in time, scaled by 1 / (W source_factor) exactly as in
        # sensitivity.py -- define_excitation supplies the remaining m -- then by k.
        # The final shift drops the leading ADJOINT_DELAY entries, which starts the
        # adjoint recursion that many steps *earlier* in its own sequence, moving
        # lambda later in physical time so it lands on the reconstructed field's
        # centre (see ADJOINT_DELAY). The tail is zero-padded, so the last step's
        # objective derivative is dropped -- lambda has barely switched on there.
        sim = self.sim
        signal = cp.asarray(dphi, dtype=sim.dtype)[::-1] / (
            self.w_sensors * sim.source_factor()
        )
        if ADJOINT_DELAY:
            signal = cp.concatenate(
                (signal[ADJOINT_DELAY:], cp.zeros_like(signal[:ADJOINT_DELAY]))
            )
        return cp.ascontiguousarray(sim.dtype(k) * signal, dtype=sim.dtype)

    def __call__(self, indicator, objective, scale=None):
        sim = self.sim
        mat = sim.build_materials(indicator)
        fd_step = define_step_method(sim, self.kernels, mat)
        excitation_step = define_excitation(
            sim, self.source.position, self.kernels, mat
        )

        U = cp.zeros((3, *sim.Nx_padded), dtype=sim.dtype)
        u0, u1, u2 = U[0], U[1], U[2]
        acc_mass = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
        acc_stiff = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
        um = cp.zeros((sim.N, self.sensors.shape[1]), dtype=sim.dtype)

        # ------------------------------- forward pass --------------------------------
        # records the traces and subtracts the forward diagonal B(u, u)
        for t in range(sim.N):
            u2 = fd_step(u0, u1, u2)
            u2 = excitation_step(u2, self.source.signal, t)
            u2 = self.bc_step(u2)
            self.get_signal(u2, um, t)
            self._accumulate(acc_mass, acc_stiff, u0, u1, u2, -1.0)
            u0, u1, u2 = u1, u2, u0

        cost, dphi = objective(um)

        k = self.scale if scale is None else scale
        self.last_scale = k
        # how big the forward diagonal is before the backward pass cancels it: the
        # ratio to what survives is the factor by which the subtraction lost
        # precision, so a k that is orders too small shows up here instead of
        # quietly returning noise
        before = float(cp.linalg.norm(acc_stiff))
        # the adjoint columns then the forward source reversed in time, so the
        # backward loop shares one row index instead of indexing N - 1 - t
        backward_signal = cp.ascontiguousarray(
            cp.concatenate(
                (self._adjoint_signal(dphi, k), self.source.signal[: sim.N][::-1]),
                axis=1,
            )
        )
        backward_excitation = define_excitation(
            sim, self.backward_position, self.kernels, mat
        )

        # ------------------------------- backward pass -------------------------------
        # u0 / u1 hold u^(N-1) / u^(N-2): the reversed recursion walks the forward
        # field back while the adjoint source is superposed on top of it, so the one
        # array carries u + k lambda and B(w, w) is added to the same accumulators
        u0, u1 = u1, u0
        for t in range(sim.N):
            u2 = fd_step(u0, u1, u2)
            u2 = backward_excitation(u2, backward_signal, t)
            u2 = self.bc_step(u2)
            self._accumulate(acc_mass, acc_stiff, u0, u1, u2, 1.0)
            u0, u1, u2 = u1, u2, u0

        # B(u, lambda) = [B(w, w) - B(u, u)] / 2k, then the same signs and cell
        # weights sensitivity.py applies: +B for the mass term, -B for the stiffness
        after = float(cp.linalg.norm(acc_stiff))
        self.last_cancellation = before / after if after > 0.0 else float("inf")
        limit = CANCELLATION_LIMIT[sim.precision]
        if self.last_cancellation > limit:
            warnings.warn(
                f"superposition scale k={k:g} leaves a cancellation of "
                f"{self.last_cancellation:.1e} in {sim.precision}, past the usable "
                f"{limit:.0e}: the gradient is largely round-off. Raise scale by "
                f"about {self.last_cancellation / limit:.0e}.",
                RuntimeWarning,
                stacklevel=2,
            )

        # scaled in place and handed over: the accumulators are private to this call,
        # and at the resolutions this class exists for a spare temporary is a field
        norm = sim.dtype(1.0 / (2.0 * k))
        apply_cell_weights(sim, acc_mass)
        acc_mass *= norm
        apply_cell_weights(sim, acc_stiff)
        acc_stiff *= -norm
        return cost, {"mass": acc_mass, "stiff": acc_stiff}, um
