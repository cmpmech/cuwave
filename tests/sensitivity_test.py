"""The adjoint variants: `sensitivity` against finite differences of the cost, and
`superposition_sensitivity` against `sensitivity`.

The reference cost goes through `simulate`, so no check reuses the adjoint module's own
forward pass. At space_order 2 the adjoint is the exact transpose, so the agreement is
round-off rather than truncation, tight enough to catch a wrong prefactor, a one-step
misalignment or a missing cell weight.

`sensitivity` is checked damped as well as lossless: the same step kernel serves both
passes there, since marching the adjoint backwards is what transposes the damped
recursion, and the finite difference is what pins the one place damping does not
cancel: the excitation sharing the update's 1 / (1 + beta) divisor. The refusal belongs
to `superposition_sensitivity` alone, whose time reversal needs a lossless operator.

The superposition variant is pinned against the exact adjoint rather than against
finite differences, so what these tests fix are the two departures it is allowed (the
symmetric Frechet form and the k^2 bias) and the property it exists for, that its
memory does not grow with N.

`source_sensitivity` differentiates the same cost with respect to the source signal
instead, so what it has to get right is a transpose rather than a bilinear form: the
cell weight at the source, the `source_factor` of whichever wave equation is running,
the damped divisor, and the time index the adjoint is read at. Each is checked against
finite differences of the signal, and once more through `point_source` and
`collect_source`, the coordinate path a driver takes.

`reconstruction_sensitivity` is allowed no departure at all: replaying the strip leaves
the reverse march reading only reversible nodes, so it is pinned to `sensitivity` at
round-off rather than at a tolerance, damped as well as lossless. What can break it is
the strip being too thin for the stencil, which shows up as a gradient error growing
with `space_order`, and a damping field that leaves nothing to rebuild from, which has
to raise.
"""

import math
import unittest
from dataclasses import replace

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # cupy missing or no GPU
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.boundary import Dirichlet, Neumann, pad_for_sponge, sponge
    from cuwave.scalar import AcousticWave, ScalarWave
    from cuwave.sensitivity import (
        l2_misfit,
        reconstruction_nodes,
        reconstruction_sensitivity,
        sensitivity,
        source_sensitivity,
        superposition_sensitivity,
    )
    from cuwave.utils import collect_source, point_source
    from cuwave.signals import ricker
    from cuwave.wave import (
        apply_cell_weights,
        Source,
        compile_kernels,
        define_excitation,
        flatten_indices,
        grid_coords,
        sensor_cell_weights,
        simulate,
        stable_dt,
    )

NC = 30  # logical grid points per axis (ghosts included)
NSTEPS = 120  # long enough for the wave to reach the sensors and reflect

# air / solid, as in examples/sensitivity/acoustic2D_sensitivity.py
RHO1, RHO2 = 1.204, 2643.0
KAPPA1, KAPPA2 = 1.419e5, 6.87e8


def _scalar(order=2, N=NSTEPS, res=NC, precision="float64", boundary=None):
    Nx, dx = (res, res - 2), (1.0 / (res - 3),) * 2
    return ScalarWave(
        Nx,
        dx,
        N,
        0.85 * stable_dt(dx, 1.0, order),
        (4, 32),
        precision=precision,
        space_order=order,
        wavespeed=1.0,
        density=1.3,
        boundary=boundary,
    )


def _acoustic(order=2, N=NSTEPS, res=NC, precision="float64"):
    Nx, dx = (res, res - 2), (1.0 / (res - 3),) * 2
    speed = math.sqrt(KAPPA2 / RHO2)
    return AcousticWave(
        Nx,
        dx,
        N,
        0.85 * stable_dt(dx, speed, order),
        (4, 32),
        precision=precision,
        space_order=order,
        rho1=RHO1,
        rho2=RHO2,
        kappa1=KAPPA1,
        kappa2=KAPPA2,
    )


def _source(sim, frequency, position):
    t = np.linspace(0, (sim.N - 1) * sim.dt, sim.N)
    signal = ricker(t, 1.0, frequency) / np.prod(sim.dx)
    return Source(
        cp.array([[position[0]], [position[1]]], dtype=cp.int32),
        cp.asarray(signal[:, None], dtype=sim.dtype),
    )


def _finite_difference(cost, field, nodes, h=1e-7):
    # central difference of the cost w.r.t. single nodes of the indicator field
    out = []
    for node in nodes:
        base = float(field[node])
        field[node] = base + h
        plus = cost()
        field[node] = base - h
        minus = cost()
        field[node] = base
        out.append((plus - minus) / (2.0 * h))
    return np.array(out)


def _damp(sim, indicator, beta):
    """Copy of `sim` damped to a uniform `beta`, whatever inertia its parametrization gives."""
    minv = sim.inverse_inertia(indicator)
    field = cp.ascontiguousarray((2.0 * beta / sim.dt / minv).astype(sim.dtype))
    return replace(sim, damping=field)


def _sponged(order=2, N=NSTEPS, res=NC, beta=0.05, precision="float64"):
    """A scalar sim grown by a five-node sponge, the layout `pad_for_sponge` builds."""
    dx = (1.0 / (res - 3),) * 2
    Nx, width, _, _ = pad_for_sponge((res, res - 2), dx, 5.0 * dx[0])
    sim = ScalarWave(
        Nx,
        dx,
        N,
        0.85 * stable_dt(dx, 1.0, order),
        (4, 32),
        precision=precision,
        space_order=order,
        wavespeed=1.0,
        density=1.3,
    )
    field = sponge(sim, cp.ones(sim.Nx_padded, dtype=sim.dtype), width, beta)
    return replace(sim, damping=field)


def _problem(sim, frequency, base):
    # wall, corner and interior, so a misweighting cannot hide in a global factor
    source = _source(sim, frequency, (1, sim.Nx[1] // 2))
    sensors = cp.array(
        [[1, sim.Nx[0] - 2, sim.Nx[0] // 3], [5, sim.Nx[1] - 2, sim.Nx[1] // 3]],
        dtype=cp.int32,
    )
    x, y = grid_coords(sim.Nx, sim.dx, dtype=sim.dtype)
    # smooth, so the material has a continuum limit and refinement means something
    indicator = cp.ascontiguousarray(
        base + 0.25 * cp.sin(2 * math.pi * x) * cp.cos(3 * math.pi * y),
        dtype=sim.dtype,
    )
    objective = l2_misfit(cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype))
    return source, sensors, indicator, objective


def _combine(sim, grads, indicator=None):
    d_mass, d_stiff = sim.parametrization_jacobian(indicator)
    return d_mass * grads["mass"] + d_stiff * grads["stiff"]


def _discrepancy(sim, a, b):
    # interior only: the Frechet form has no wall closure, so the ring differs most
    sl = tuple(slice(2, n - 2) for n in sim.Nx)
    a, b = a[sl], b[sl]
    rel = float(cp.linalg.norm(a - b) / cp.linalg.norm(b))
    cos = float(cp.sum(a * b) / (cp.linalg.norm(a) * cp.linalg.norm(b)))
    return rel, cos


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class GradientTest(unittest.TestCase):
    def _nodes(self, sim):
        # a wall node per axis, a corner, interior: the ring pins the cell weights
        hi0, hi1 = sim.Nx[0] - 2, sim.Nx[1] - 2
        return [(1, 12), (hi0, 12), (1, 1), (11, 9), (17, 14), (2, 7), (8, hi1)]

    def _sensors(self, sim):
        # wall, corner and interior, so a misweighting cannot hide in a global factor
        hi0, hi1 = sim.Nx[0] - 2, sim.Nx[1] - 2
        return cp.array([[1, hi0, hi0, 11], [5, hi1, 9, hi1]], dtype=cp.int32)

    def _check(self, sim, indicator, objective, source, sensors, tol=2e-6, nodes=None):
        cost, grads, traces, _ = sensitivity(sim, source, indicator, sensors, objective)

        def cost_of():
            return objective(simulate(sim, source, indicator, sensors=sensors)[1])[0]

        # a gradient check against a cost of ~0 agrees with anything
        self.assertGreater(
            abs(cost_of()), 1e-9, "the sensors record essentially nothing"
        )
        self.assertAlmostEqual(cost / cost_of(), 1.0, places=10, msg="forward mismatch")

        gradient = _combine(sim, grads, indicator)

        nodes = nodes or self._nodes(sim)
        reference = _finite_difference(cost_of, indicator, nodes)
        adjoint = np.array([float(gradient[n]) for n in nodes])
        scale = max(np.max(np.abs(reference)), 1e-30)
        error = np.max(np.abs(adjoint - reference)) / scale
        self.assertLess(error, tol, f"gradient off by {error:.2e}")
        return traces

    def _random(self, sim, base, spread):
        cp.random.seed(0)
        return (base + spread * cp.random.rand(*sim.Nx_padded)).astype(sim.dtype)

    def test_scalar_gradient_matches_finite_differences(self):
        sim = _scalar()
        indicator = self._random(sim, 1.0, 0.3)
        source = _source(sim, 4.0, (1, (NC - 2) // 2))
        sensors = self._sensors(sim)
        observed = simulate(sim, source, indicator, sensors=sensors)[1] * 0.7
        self._check(sim, indicator, l2_misfit(observed), source, sensors)

    def test_acoustic_gradient_matches_finite_differences(self):
        # a different linear combination of the two gradients than the scalar case
        sim = _acoustic()
        indicator = self._random(sim, 0.3, 0.3)
        source = _source(sim, 400.0, (1, (NC - 2) // 2))
        sensors = self._sensors(sim)
        observed = simulate(sim, source, indicator, sensors=sensors)[1] * 0.7
        self._check(sim, indicator, l2_misfit(observed), source, sensors)

    def test_damped_gradient_matches_finite_differences(self):
        # a damping proportional to the inertia, so both formulations get the same beta
        beta = 0.02
        for name, lossless_sim, base, hz in (
            ("scalar", _scalar(), 1.0, 4.0),
            ("acoustic", _acoustic(), 0.3, 400.0),
        ):
            with self.subTest(formulation=name):
                indicator = self._random(lossless_sim, base, 0.3)
                source = _source(lossless_sim, hz, (1, (NC - 2) // 2))
                sensors = self._sensors(lossless_sim)
                sim = _damp(lossless_sim, indicator, beta)
                quiet = simulate(sim, source, indicator, sensors=sensors)[1]
                loud = simulate(lossless_sim, source, indicator, sensors=sensors)[1]
                decay = float(cp.linalg.norm(quiet) / cp.linalg.norm(loud))
                print(
                    f"  {name}: beta {beta}, traces at {decay:.3f} of the lossless run"
                )
                self.assertLess(
                    decay, 0.5, "the damping barely changed the forward run"
                )
                self._check(sim, indicator, l2_misfit(quiet * 0.7), source, sensors)

    def test_dirichlet_wall_gradient_matches_finite_differences(self):
        # the cell weights need no Dirichlet variant; (1, 12) still probes the layer
        sim = _scalar(boundary=((Dirichlet, Neumann), (Neumann, Neumann)))
        self.assertEqual(sim.boundary[0][0], Dirichlet)
        indicator = self._random(sim, 1.0, 0.3)
        # off the Dirichlet layer, where a source would drive nothing
        source = _source(sim, 4.0, (5, (NC - 2) // 2))
        hi0, hi1 = sim.Nx[0] - 2, sim.Nx[1] - 2
        sensors = cp.array([[3, hi0, hi0, 11], [5, hi1, 9, hi1]], dtype=cp.int32)
        observed = simulate(sim, source, indicator, sensors=sensors)[1] * 0.7
        self._check(
            sim,
            indicator,
            l2_misfit(observed),
            source,
            sensors,
            nodes=[(1, 12), (2, 7), (hi0, 12), (11, 9), (8, hi1)],
        )

    def test_arbitrary_objective(self):
        # any differentiable cost works: only its derivative reaches the adjoint
        sim = _scalar()
        indicator = self._random(sim, 1.0, 0.3)
        source = _source(sim, 4.0, (7, (NC - 2) // 2))
        sensors = self._sensors(sim)
        weight = cp.asarray(
            np.random.RandomState(0).rand(sim.N, sensors.shape[1]), dtype=sim.dtype
        )

        def objective(traces):
            return float(cp.sum(weight * traces**3)), 3.0 * weight * traces**2

        self._check(sim, indicator, objective, source, sensors)

    def test_duplicate_excitation_nodes_all_land(self):
        # excitation_kernel uses atomicAdd, so two entries on one node both land
        sim = _scalar(N=1)
        node = (3, 4)
        num = 8  # the same node, several times over, inside one launch
        position = cp.array([[node[0]] * num, [node[1]] * num], dtype=cp.int32)
        signal = cp.asarray(
            np.arange(1, num + 1, dtype=float)[None, :], dtype=sim.dtype
        )
        indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
        mat = sim.build_materials(indicator)
        kernels = compile_kernels(sim)
        excitation = define_excitation(sim, position, kernels, mat)

        u = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
        excitation(u, signal, 0)
        # one shared weight over every column, so the total is weight * sum(1..num)
        weight = float(sim.excitation_weights(mat, flatten_indices(sim, position))[0])
        expected = weight * num * (num + 1) / 2
        self.assertAlmostEqual(float(u[node]) / expected, 1.0, places=10)
        self.assertEqual(float(cp.count_nonzero(u)), 1.0, "spilled onto other nodes")

    def test_ghost_material_is_slaved_to_its_mirror(self):
        # build_materials mirrors the ghost ring in place, so the gradient there is 0
        sim = _scalar()
        indicator = self._random(sim, 1.0, 0.3)
        source = _source(sim, 4.0, (1, (NC - 2) // 2))
        sensors = self._sensors(sim)
        objective = l2_misfit(cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype))

        _, grads, _, _ = sensitivity(sim, source, indicator, sensors, objective)
        gradient = _combine(sim, grads, indicator)
        for ghost in ((0, 12), (sim.Nx[0] - 1, 12), (12, 0), (12, sim.Nx[1] - 1)):
            self.assertEqual(float(gradient[ghost]), 0.0)

        def cost_of():
            return objective(simulate(sim, source, indicator, sensors=sensors)[1])[0]

        reference = _finite_difference(cost_of, indicator, [(0, 12), (12, 0)], h=1e-3)
        np.testing.assert_allclose(reference, 0.0, atol=1e-12)

        # and the in-place normalisation leaves the ghost equal to its mirror
        self.assertEqual(float(indicator[0, 12]), float(indicator[2, 12]))
        self.assertEqual(float(indicator[12, 0]), float(indicator[12, 2]))

    def test_the_two_cell_weight_paths_agree(self):
        # W is never materialised, so its two paths must agree node for node
        for sim in (_scalar(), _acoustic()):
            with self.subTest(ndim=sim.ndim):
                field = apply_cell_weights(sim, cp.ones(sim.Nx_padded, dtype=sim.dtype))
                hi0, hi1 = sim.Nx[0] - 2, sim.Nx[1] - 2
                nodes = [(1, 1), (1, 12), (hi0, hi1), (hi0, 12), (12, 1), (11, 9)]
                probe = cp.array(
                    [[i for i, _ in nodes], [j for _, j in nodes]], dtype=cp.int32
                )
                expected = np.array([float(field[n]) for n in nodes])
                np.testing.assert_allclose(
                    sensor_cell_weights(sim, probe).get(), expected, rtol=0, atol=0
                )
                # and the values are the {1, 1/2, 1/4} the derivation calls for
                self.assertEqual(expected[0], 0.25)
                self.assertEqual(expected[1], 0.5)
                self.assertEqual(expected[-1], 1.0)

    def test_high_order_is_only_consistent(self):
        # above order 2 the transpose is symmetric to O(dx^2), on a smooth material
        sim = _scalar(order=4)
        x, y = grid_coords(sim.Nx, sim.dx, dtype=sim.dtype)
        indicator = cp.ascontiguousarray(
            1.0 + 0.25 * cp.sin(3.0 * math.pi * x) * cp.cos(2.0 * math.pi * y),
            dtype=sim.dtype,
        )
        source = _source(sim, 4.0, (7, (NC - 2) // 2))
        sensors = cp.array([[20, 11], [7, 20]], dtype=cp.int32)
        observed = simulate(sim, source, indicator, sensors=sensors)[1] * 0.7
        self._check(
            sim,
            indicator,
            l2_misfit(observed),
            source,
            sensors,
            tol=5e-2,
            nodes=[(11, 9), (17, 14), (13, 11), (9, 15)],
        )


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class SuperpositionTest(unittest.TestCase):
    def _both(self, sim, frequency, base, scale=1.0):
        source, sensors, indicator, objective = _problem(sim, frequency, base)
        reference = sensitivity(sim, source, indicator, sensors, objective)
        superposed = superposition_sensitivity(
            sim, source, indicator, sensors, objective, scale=scale
        )
        return reference, superposed, sim

    def _cases(self):
        return (
            ("scalar", _scalar(N=150, res=40), 5.0, 1.0),
            ("acoustic", _acoustic(N=150, res=40), 400.0, 0.4),
        )

    def test_cost_and_traces_match_exactly(self):
        # the same forward simulation, so only the gradient path differs
        for name, sim, freq, base in self._cases():
            with self.subTest(formulation=name):
                (c0, _, t0, _), (c1, _, t1, _), _ = self._both(sim, freq, base)
                self.assertEqual(c0, c1)
                self.assertTrue(bool(cp.all(t0 == t1)))

    def test_gradient_agrees_with_the_exact_adjoint(self):
        # a different discretisation of the same sensitivity, so directions agree
        for name, sim, freq, base in self._cases():
            with self.subTest(formulation=name):
                (_, gr, _, _), (_, g, _, _), _ = self._both(sim, freq, base)
                rel, cos = _discrepancy(sim, _combine(sim, g), _combine(sim, gr))
                self.assertLess(rel, 0.25, f"{name}: relative difference {rel:.3e}")
                self.assertGreater(cos, 0.98, f"{name}: cosine {cos:.6f}")

    def test_it_is_a_descent_direction(self):
        # what an optimizer needs: stepping against it must reduce the forward cost
        sim = _scalar(N=150, res=40)
        source, sensors, indicator, objective = _problem(sim, 5.0, 1.0)
        cost, grads, _, _ = superposition_sensitivity(
            sim, source, indicator, sensors, objective
        )
        gradient = _combine(sim, grads)

        def cost_of(field):
            return objective(simulate(sim, source, field, sensors=sensors)[1])[0]

        step = 1e-3 / float(cp.max(cp.abs(gradient)))
        self.assertLess(cost_of(indicator - step * gradient), cost)
        # and the directional derivative has the sign the gradient claims
        forward = (cost_of(indicator + step * gradient) - cost) / step
        self.assertGreater(forward, 0.0)

    def test_it_converges_to_the_exact_adjoint(self):
        # a discretization difference, so it has to shrink under refinement
        errors = []
        for res, steps in ((40, 74), (80, 154)):
            (_, gr, _, _), (_, g, _, _), sim = self._both(
                _scalar(N=steps, res=res), 5.0, 1.0
            )
            rel, _ = _discrepancy(sim, _combine(sim, g), _combine(sim, gr))
            errors.append(rel)
        self.assertLess(errors[1], 0.6 * errors[0], f"not converging: {errors}")

    def test_cancellation_diagnostic_tracks_the_scale(self):
        # the cancellation must scale as 1/k, which is what makes a bad scale visible
        sim = _scalar(N=150, res=40)
        source, sensors, indicator, objective = _problem(sim, 5.0, 1.0)
        args = (sim, source, indicator, sensors, objective)
        loose = superposition_sensitivity(*args, scale=1.0)[3]["cancellation"]
        tight = superposition_sensitivity(*args, scale=1e-2)[3]["cancellation"]
        self.assertAlmostEqual(tight / loose, 100.0, delta=1.0)

    def test_memory_does_not_grow_with_the_number_of_steps(self):
        # the whole point: the standard adjoint stores N + 2 fields, this stores 3
        pool = cp.get_default_memory_pool()
        peaks = {}
        # big enough that the grids dominate the (N, sensors) arrays
        for steps in (100, 800):
            for name, run in (
                ("standard", sensitivity),
                ("superposition", superposition_sensitivity),
            ):
                pool.free_all_blocks()
                sim = _scalar(N=steps, res=192, precision="float32")
                source, sensors, indicator, objective = _problem(sim, 5.0, 1.0)
                run(sim, source, indicator, sensors, objective)
                peaks[name, steps] = pool.total_bytes()
                pool.free_all_blocks()

        # 8x the steps must not measurably move the superposition footprint...
        self.assertLess(peaks["superposition", 800] / peaks["superposition", 100], 1.2)
        # ...while it moves the standard one by close to the same factor
        self.assertGreater(peaks["standard", 800] / peaks["standard", 100], 5.0)

    def test_a_sensor_on_the_source_still_gives_the_right_gradient(self):
        # a co-located source and receiver: two contributions, one node, one launch
        sim = _scalar(N=150, res=40)
        source, sensors, indicator, _ = _problem(sim, 5.0, 1.0)
        shared = cp.concatenate((source.position, sensors[:, 2:]), axis=1)
        objective = l2_misfit(cp.zeros((sim.N, shared.shape[1]), dtype=sim.dtype))

        _, grads, _, _ = superposition_sensitivity(
            sim, source, indicator, shared, objective
        )
        _, grads_ref, _, _ = sensitivity(sim, source, indicator, shared, objective)
        rel, cos = _discrepancy(sim, _combine(sim, grads), _combine(sim, grads_ref))
        self.assertLess(rel, 0.25, f"relative difference {rel:.3e}")
        self.assertGreater(cos, 0.98, f"cosine {cos:.6f}")

    def test_damping_is_refused(self):
        sim = _scalar(N=150, res=40)
        source, sensors, indicator, objective = _problem(sim, 5.0, 1.0)
        with self.assertRaises(NotImplementedError):
            superposition_sensitivity(
                _damp(sim, indicator, 0.02), source, indicator, sensors, objective
            )

    def test_sensors_on_a_ghost_node_are_refused(self):
        sim = _scalar(N=150, res=40)
        source, _, indicator, objective = _problem(sim, 5.0, 1.0)
        ghost = cp.array([[sim.Nx[0] - 1], [5]], dtype=cp.int32)
        with self.assertRaises(ValueError):
            superposition_sensitivity(sim, source, indicator, ghost, objective)


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class ReconstructionTest(unittest.TestCase):
    def _both(self, sim):
        source, sensors, indicator, objective = _problem(sim, 5.0, 1.0)
        reference = sensitivity(sim, source, indicator, sensors, objective)
        rebuilt = reconstruction_sensitivity(sim, source, indicator, sensors, objective)
        _, valid = reconstruction_nodes(sim)
        return reference, rebuilt, valid

    def _cases(self):
        return (("lossless", _scalar()), ("sponged", _sponged()))

    def test_cost_and_traces_match_sensitivity(self):
        # the same kernel over the same slots, so the forward values are the same bits
        for name, sim in self._cases():
            with self.subTest(operator=name):
                (c0, _, t0, _), (c1, _, t1, _), _ = self._both(sim)
                self.assertEqual(c0, c1)
                self.assertTrue(bool(cp.all(t0 == t1)))

    def test_the_gradient_is_the_exact_transpose(self):
        for name, sim in self._cases():
            with self.subTest(operator=name):
                (_, gr, _, _), (_, g, _, _), valid = self._both(sim)
                a, b = _combine(sim, g)[valid], _combine(sim, gr)[valid]
                rel = float(cp.linalg.norm(a - b) / cp.linalg.norm(b))
                self.assertLess(rel, 1e-12, f"{name}: relative difference {rel:.3e}")

    def test_the_strip_is_thick_enough_for_a_wide_stencil(self):
        # too thin a strip lets an irreversible node into the stencil, and only there
        for order in (2, 4, 6):
            with self.subTest(space_order=order):
                sim = _sponged(order=order)
                (_, gr, _, _), (_, g, _, _), valid = self._both(sim)
                a, b = _combine(sim, g)[valid], _combine(sim, gr)[valid]
                rel = float(cp.linalg.norm(a - b) / cp.linalg.norm(b))
                self.assertLess(rel, 1e-12, f"order {order}: difference {rel:.3e}")

    def test_valid_covers_the_whole_design_region(self):
        # comparing on `valid` is self-referential, so pin `valid` against `region`
        dx = (1.0 / (NC - 3),) * 2
        region = pad_for_sponge((NC, NC - 2), dx, 5.0 * dx[0])[3]
        for order in (2, 4):
            with self.subTest(space_order=order):
                _, valid = reconstruction_nodes(_sponged(order=order))
                design = cp.zeros(valid.shape, dtype=cp.bool_)
                design[region] = True
                dropped = int((design & ~valid).sum())
                self.assertEqual(dropped, 0, f"{dropped} design nodes outside valid")

    def test_it_matches_finite_differences_through_a_sponge(self):
        # the check no other variant runs damped: superposition refuses the field
        sim = _sponged()
        source, sensors, indicator, objective = _problem(sim, 5.0, 1.0)
        _, grads, _, _ = reconstruction_sensitivity(
            sim, source, indicator, sensors, objective
        )
        nodes = [(11, 9), (17, 14), (13, 11), (22, 15)]

        def cost():
            return objective(simulate(sim, source, indicator, sensors=sensors)[1])[0]

        reference = _finite_difference(cost, indicator, nodes)
        gradient = _combine(sim, grads)
        for node, expected in zip(nodes, reference):
            self.assertAlmostEqual(
                float(gradient[node]) / expected, 1.0, delta=1e-4, msg=f"node {node}"
            )

    def test_a_closed_lossless_domain_needs_no_strip(self):
        # nothing is irreversible, so the reverse march runs on the two seeds alone
        sim = _scalar()
        (_, gr, _, _), (_, g, _, info), _ = self._both(sim)
        self.assertEqual(info["strip"], 0)
        rel, _ = _discrepancy(sim, _combine(sim, g), _combine(sim, gr))
        self.assertLess(rel, 1e-12, f"relative difference {rel:.3e}")

    def test_memory_grows_with_the_strip_not_the_grid(self):
        # the point: the strip is a ring of nodes where the history is a grid of them
        pool = cp.get_default_memory_pool()
        peaks = {}
        for steps in (100, 800):
            for name, run in (
                ("standard", sensitivity),
                ("reconstruction", reconstruction_sensitivity),
            ):
                pool.free_all_blocks()
                sim = _sponged(N=steps, res=192, precision="float32")
                source, sensors, indicator, objective = _problem(sim, 5.0, 1.0)
                run(sim, source, indicator, sensors, objective)
                peaks[name, steps] = pool.total_bytes()
                pool.free_all_blocks()

        self.assertLess(peaks["reconstruction", 800] / peaks["standard", 800], 0.1)
        # 8x the steps moves the ring far less than it moves the grid
        grew = peaks["reconstruction", 800] / peaks["reconstruction", 100]
        self.assertLess(grew, 0.5 * peaks["standard", 800] / peaks["standard", 100])

    def test_a_fully_damped_simulation_is_refused(self):
        # a uniform field leaves no lossless node, so there is nothing to rebuild from
        sim = _scalar()
        source, sensors, indicator, objective = _problem(sim, 5.0, 1.0)
        with self.assertRaises(ValueError):
            reconstruction_sensitivity(
                _damp(sim, indicator, 0.02), source, indicator, sensors, objective
            )


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class SourceGradientTest(unittest.TestCase):
    def _nodes(self, sim):
        # wall, corner and interior, so a misweighting cannot hide in a global factor
        hi0, hi1 = sim.Nx[0] - 2, sim.Nx[1] - 2
        return cp.array([[1, hi0, 9], [7, hi1, 11]], dtype=cp.int32)

    def _sensors(self, sim):
        hi0, hi1 = sim.Nx[0] - 2, sim.Nx[1] - 2
        return cp.array([[1, hi0, hi0, 11], [5, hi1, 9, hi1]], dtype=cp.int32)

    def _shot(self, sim, frequency):
        t = np.linspace(0, (sim.N - 1) * sim.dt, sim.N)
        signal = ricker(t, 1.0, frequency) / np.prod(sim.dx)
        # a scaling per column, so one column cannot stand in for another
        columns = signal[:, None] * np.array([1.0, 0.4, -0.7])
        position = self._nodes(sim)
        return Source(position, cp.asarray(columns, dtype=sim.dtype))

    def _check(self, sim, indicator, objective, source, sensors, tol=1e-8):
        cost, gradient, _, _ = source_sensitivity(
            sim, source, indicator, sensors, objective
        )

        def cost_of():
            return objective(simulate(sim, source, indicator, sensors=sensors)[1])[0]

        self.assertGreater(
            abs(cost_of()), 1e-9, "the sensors record essentially nothing"
        )
        self.assertAlmostEqual(cost / cost_of(), 1.0, places=10, msg="forward mismatch")

        # late steps are excluded: their adjoint has not reached the source yet
        entries = [(0, 0), (4, 1), (11, 2), (30, 0), (55, 1), (70, 2)]
        reference = _finite_difference(cost_of, source.signal, entries, h=1e-2)
        adjoint = np.array([float(gradient[e]) for e in entries])
        scale = max(np.max(np.abs(reference)), 1e-30)
        error = np.max(np.abs(adjoint - reference)) / scale
        self.assertLess(error, tol, f"source gradient off by {error:.2e}")

    def _random(self, sim, base, spread):
        cp.random.seed(0)
        return (base + spread * cp.random.rand(*sim.Nx_padded)).astype(sim.dtype)

    def test_scalar_source_gradient_matches_finite_differences(self):
        sim = _scalar()
        indicator = self._random(sim, 1.0, 0.3)
        source = self._shot(sim, 4.0)
        sensors = self._sensors(sim)
        observed = simulate(sim, source, indicator, sensors=sensors)[1] * 0.7
        self._check(sim, indicator, l2_misfit(observed), source, sensors)

    def test_acoustic_source_gradient_matches_finite_differences(self):
        # the one thing the parametrization changes here is source_factor
        sim = _acoustic()
        indicator = self._random(sim, 0.3, 0.3)
        source = self._shot(sim, 400.0)
        sensors = self._sensors(sim)
        observed = simulate(sim, source, indicator, sensors=sensors)[1] * 0.7
        self._check(sim, indicator, l2_misfit(observed), source, sensors)

    def test_damped_source_gradient_matches_finite_differences(self):
        for name, lossless_sim, base, hz in (
            ("scalar", _scalar(), 1.0, 4.0),
            ("acoustic", _acoustic(), 0.3, 400.0),
        ):
            with self.subTest(formulation=name):
                indicator = self._random(lossless_sim, base, 0.3)
                source = self._shot(lossless_sim, hz)
                sensors = self._sensors(lossless_sim)
                sim = _damp(lossless_sim, indicator, 0.02)
                quiet = simulate(sim, source, indicator, sensors=sensors)[1]
                self._check(sim, indicator, l2_misfit(quiet * 0.7), source, sensors)

    def test_the_coordinate_path_matches_finite_differences(self):
        # point_source and collect_source, the pair a driver differentiates through
        sim = _scalar()
        indicator = self._random(sim, 1.0, 0.3)
        sensors = self._sensors(sim)
        coords = [(0.31, 0.42)]  # off-node, so every corner weight is exercised
        t = np.linspace(0, (sim.N - 1) * sim.dt, sim.N)
        signal = cp.asarray(ricker(t, 1.0, 4.0), dtype=sim.dtype)
        observed = (
            simulate(
                sim, point_source(sim, coords, signal), indicator, sensors=sensors
            )[1]
            * 0.7
        )
        objective = l2_misfit(observed)

        def cost_of():
            source = point_source(sim, coords, signal)
            return objective(simulate(sim, source, indicator, sensors=sensors)[1])[0]

        _, columns, _, _ = source_sensitivity(
            sim, point_source(sim, coords, signal), indicator, sensors, objective
        )
        gradient = collect_source(sim, coords, columns)

        steps = [0, 4, 11, 30, 55, 70]
        reference = _finite_difference(cost_of, signal, steps, h=1e-5)
        adjoint = np.array([float(gradient[n, 0]) for n in steps])
        scale = max(np.max(np.abs(reference)), 1e-30)
        error = np.max(np.abs(adjoint - reference)) / scale
        self.assertLess(error, 1e-8, f"coordinate gradient off by {error:.2e}")

    def test_a_source_on_a_ghost_node_is_refused(self):
        sim = _scalar()
        indicator = self._random(sim, 1.0, 0.3)
        sensors = self._sensors(sim)
        source = Source(
            cp.array([[0], [9]], dtype=cp.int32),
            cp.zeros((sim.N, 1), dtype=sim.dtype),
        )
        with self.assertRaises(ValueError):
            source_sensitivity(sim, source, indicator, sensors, l2_misfit(0.0))


if __name__ == "__main__":
    unittest.main()
