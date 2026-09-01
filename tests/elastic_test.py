"""Contracts of the staggered elastic equation that the scalar tests cannot reach.

Three of them carry the rest. The staggered strains put every stress on a natural point,
so the operator must converge at its full order while the cost stays flat in the radius.
The operator is `-B^T C B` with matching taps in both kernels, so it must be exactly
symmetric at every order and boundary mix, which is what all adjoint variants rest on.
And the staggering shifts every component off the nodes, so sources, sensors and the
reconstruction strip must all honour the half-node offsets.
"""

import unittest
from dataclasses import replace

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.boundary import Clamped, Traction, pad_for_sponge, sponge
    from cuwave.elastic import ElasticWave, stable_timestep
    from cuwave.scalar import ScalarWave
    from cuwave.sensitivity import (
        l2_misfit,
        reconstruction_nodes,
        reconstruction_sensitivity,
        sensitivity,
        source_sensitivity,
        superposition_sensitivity,
    )
    from cuwave.signals import ricker
    from cuwave.stencils import staggered_weights
    from cuwave.utils import Sensors, point_source
    from cuwave.wave import (
        Source,
        compile_kernels,
        define_step_method,
        simulate,
        stable_dt,
    )

CP, CS, RHO = 2.0, 1.0, 1.3
THREADS = {1: (128,), 2: (8, 8), 3: (4, 4, 8)}


# -------------------------------------- helpers --------------------------------------
def _sim(Nx, N=80, precision="float64", space_order=2, **kwargs):
    dx = tuple(1.0 / (n - 3) for n in Nx)
    return ElasticWave(
        Nx,
        dx,
        N,
        0.8 * stable_dt(dx, CP, space_order),
        THREADS[len(Nx)][::-1],
        precision=precision,
        space_order=space_order,
        density=RHO,
        wavespeed_p=CP,
        wavespeed_s=CS,
        **kwargs,
    )


def _problem(sim, seed=0):
    """A vector shot, a two component receiver pair and a rough starting model."""
    rng = np.random.default_rng(seed)
    t = np.arange(sim.N) * sim.dt
    signal = ricker(t, 1.0, 6.0)
    mid = [n // 2 for n in sim.Nx]
    source = point_source(
        sim,
        [[(m - 2) * h for m, h in zip(mid, sim.dx)]],
        signal,
        direction=[0.0] * (sim.ncomp - 1) + [1.0],
    )
    sensors = cp.asarray(
        [[c for c in range(sim.ncomp)]] + [[m + 2] * sim.ncomp for m in mid],
        dtype=cp.int32,
    )
    indicator = cp.asarray(0.6 + 0.3 * rng.random(sim.Nx_padded), dtype=sim.dtype)
    objective = l2_misfit(cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype))
    return source, sensors, indicator, objective


def _operator(sim, indicator):
    """The mass weighted spatial operator on the free unknowns, an exact transpose."""
    mat = sim.build_materials(indicator)
    step = define_step_method(sim, compile_kernels(sim), mat)
    mass = cp.where(mat["minv"] > 0, 1.0 / cp.maximum(mat["minv"], 1e-300), 0.0)
    slices = [
        (c, *(slice(1, n - 1 - (d == c)) for d, n in enumerate(sim.Nx)))
        for c in range(sim.ncomp)
    ]

    def mask(x):
        out = cp.zeros_like(x)
        for sl in slices:
            out[sl] = x[sl]
        return out * (mat["minv"] > 0)

    def apply(x):
        out = cp.zeros_like(x)
        step(cp.zeros_like(x), x, out)
        return mask((2.0 * x - out) * mass)

    return apply, mask


def _finite_difference(cost_of, indicator, nodes, h=1e-6):
    """Central difference of the cost at each of `nodes`."""
    out = []
    for node in nodes:
        plus, minus = indicator.copy(), indicator.copy()
        plus[node] += h
        minus[node] -= h
        out.append((cost_of(plus) - cost_of(minus)) / (2.0 * h))
    return np.array(out)


def _divergence_error(order, n):
    """Relative operator error against the analytic div(sigma) of a smooth field."""
    lame, shear = 1.0, 0.6
    a, b, c, e = 2.0 * np.pi, 3.0 * np.pi, np.pi, 2.0 * np.pi
    speed_p, speed_s = np.sqrt((lame + 2.0 * shear) / RHO), np.sqrt(shear / RHO)
    dx = (1.0 / n, 1.3 / n)
    sim = ElasticWave(
        (n + 3, n + 3),
        dx,
        1,
        0.9 * stable_dt(dx, speed_p, order),
        (8, 32),
        precision="float64",
        space_order=order,
        density=RHO,
        wavespeed_p=speed_p,
        wavespeed_s=speed_s,
    )
    axes = [(np.arange(m) - 1) * h for m, h in zip(sim.Nx_padded, dx)]
    X, Y = np.meshgrid(*axes, indexing="ij")
    # each component sampled at its own half-shifted point
    Xs, Ys = X + 0.5 * dx[0], Y + 0.5 * dx[1]
    u = np.stack([np.sin(a * Xs) * np.cos(b * Y), np.cos(c * X) * np.sin(e * Ys)])
    gx = -(a * a) * np.sin(a * Xs) * np.cos(b * Y) - e * c * np.sin(c * Xs) * np.cos(
        e * Y
    )
    gy = -a * b * np.cos(a * X) * np.sin(b * Ys) - (e * e) * np.cos(c * X) * np.sin(
        e * Ys
    )
    lx = -(a * a + b * b) * np.sin(a * Xs) * np.cos(b * Y)
    ly = -(c * c + e * e) * np.cos(c * X) * np.sin(e * Ys)
    want = np.stack(
        [(lame + shear) * gx + shear * lx, (lame + shear) * gy + shear * ly]
    )
    u1 = cp.asarray(u, dtype=sim.dtype)
    mat = sim.build_materials(cp.ones(sim.Nx_padded, dtype=sim.dtype))
    step = define_step_method(sim, compile_kernels(sim), mat)
    u0, u2 = cp.zeros_like(u1), cp.zeros_like(u1)
    step(u0, u1, u2)
    cp.cuda.Stream.null.synchronize()
    got = cp.asnumpy(u2 - 2.0 * u1) * RHO / sim.dt**2
    margin = order + 1
    inner = (Ellipsis, slice(margin, n - margin), slice(margin, n - margin))
    return np.abs(got[inner] - want[inner]).max() / np.abs(want[inner]).max()


# -------------------------------- stencil and operator -------------------------------
class StencilTest(unittest.TestCase):
    def test_the_weights_are_the_staggered_taylor_ones(self):
        np.testing.assert_allclose(staggered_weights(1), [1.0])
        np.testing.assert_allclose(staggered_weights(2), [9.0 / 8.0, -1.0 / 24.0])

    def test_the_weights_differentiate_at_their_order(self):
        for r in (1, 2, 3):
            errors = []
            for h in (0.02, 0.01):
                c = staggered_weights(r)
                taps = sum(
                    c[k - 1] * (np.sin((k - 0.5) * h) - np.sin(-(k - 0.5) * h))
                    for k in range(1, r + 1)
                )
                errors.append(abs(taps / h - 1.0))
            self.assertGreater(np.log2(errors[0] / errors[1]), 2 * r - 0.5)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class OperatorTest(unittest.TestCase):
    def test_the_stencil_converges_at_its_order(self):
        for order in (2, 4, 6):
            with self.subTest(order=order):
                errors = [_divergence_error(order, n) for n in (32, 64)]
                rate = np.log2(errors[0] / errors[1])
                self.assertGreater(rate, order - 0.5)

    def test_the_operator_is_symmetric_at_every_order_and_boundary(self):
        rng = np.random.default_rng(4)
        cases = (
            ((18, 20), 2, None),
            ((18, 20), 4, None),
            ((18, 20), 6, Clamped),
            ((18, 20), 4, ((Traction, Clamped), (Clamped, Traction))),
            ((12, 11, 13), 4, Clamped),
        )
        for Nx, order, boundary in cases:
            with self.subTest(Nx=Nx, order=order, boundary=boundary):
                sim = _sim(Nx, N=1, space_order=order, boundary=boundary)
                gamma = cp.asarray(0.4 + rng.random(sim.Nx_padded), dtype=sim.dtype)
                apply, mask = _operator(sim, gamma)
                a = mask(
                    cp.asarray(
                        rng.standard_normal((sim.ncomp, *sim.Nx_padded)),
                        dtype=sim.dtype,
                    )
                )
                b = mask(
                    cp.asarray(
                        rng.standard_normal((sim.ncomp, *sim.Nx_padded)),
                        dtype=sim.dtype,
                    )
                )
                lhs = float(cp.sum(b * apply(a)))
                rhs = float(cp.sum(a * apply(b)))
                self.assertLess(abs(lhs - rhs), 1e-12 * abs(lhs))

    def test_the_checkerboard_is_not_a_null_mode(self):
        sim = _sim((18, 20), N=1)
        gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
        apply, mask = _operator(sim, gamma)
        axes = cp.meshgrid(
            *[cp.arange(n, dtype=sim.dtype) for n in sim.Nx_padded], indexing="ij"
        )
        checker = (-1.0) ** (axes[0] + axes[1])
        field = cp.zeros((sim.ncomp, *sim.Nx_padded), dtype=sim.dtype)
        field[:] = checker
        # a staggered difference does not annihilate the alternating mode
        masked = mask(field)
        self.assertGreater(
            float(cp.linalg.norm(apply(masked))), 1e-3 * float(cp.linalg.norm(masked))
        )

    def test_the_cost_stays_flat_in_the_order(self):
        import time

        times = {}
        for order in (2, 6):
            sim = _sim((260, 260), N=1, precision="float32", space_order=order)
            mat = sim.build_materials(cp.ones(sim.Nx_padded, dtype=sim.dtype))
            step = define_step_method(sim, compile_kernels(sim), mat)
            u = cp.zeros((3, sim.ncomp, *sim.Nx_padded), dtype=sim.dtype)
            for _ in range(5):
                step(u[0], u[1], u[2])
            cp.cuda.Stream.null.synchronize()
            tic = time.time()
            for _ in range(50):
                step(u[0], u[1], u[2])
            cp.cuda.Stream.null.synchronize()
            times[order] = time.time() - tic
        # the cell gather paid 76x here; per-axis staggering pays a small factor
        self.assertLess(times[6], 3.0 * times[2])


# ---------------------------------- against the scalar -------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class OneDimensionTest(unittest.TestCase):
    """On a uniform medium the 1D staggered scheme is the scalar stencil on the offset
    grid, so the two must agree to round-off while no reflection is in flight."""

    def test_it_reproduces_the_scalar_solver_before_the_walls_answer(self):
        res, N = 400, 150
        Nx, dx = (res,), (1.0 / (res - 3),)
        dt = 0.7 * stable_dt(dx, CP, 2)
        common = dict(precision="float64", space_order=2)
        scalar = ScalarWave(Nx, dx, N, dt, (128,), wavespeed=CP, density=RHO, **common)
        elastic = ElasticWave(
            Nx,
            dx,
            N,
            dt,
            (128,),
            density=RHO,
            wavespeed_p=CP,
            wavespeed_s=0.8 * CP,
            **common,
        )
        t = np.arange(N) * dt
        signal = ricker(t, 1.0, 12.0)
        volume = float(np.prod(dx))
        gamma = 0.8
        # the same discrete system: node i of the scalar is unknown i of the staggered
        a = simulate(
            scalar,
            point_source(scalar, [[139 * dx[0]]], signal),
            cp.full(scalar.Nx_padded, gamma),
        )
        b = simulate(
            elastic,
            point_source(elastic, [[139.5 * dx[0]]], signal * volume, direction=[1.0]),
            cp.full(elastic.Nx_padded, gamma),
        )
        a, b = cp.asnumpy(a)[1:-1], cp.asnumpy(b)[1:-2]
        self.assertLess(np.abs(a[:-1] - b).max(), 1e-12 * np.abs(a).max())


# ------------------------------------- sensitivity -----------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class GradientTest(unittest.TestCase):
    def _check(self, sim, nodes, tol=1e-4):
        source, sensors, indicator, objective = _problem(sim)

        def cost_of(field):
            return objective(simulate(sim, source, field, sensors=sensors)[1])[0]

        cost, grads, _, _ = sensitivity(sim, source, indicator, sensors, objective)
        self.assertAlmostEqual(cost / cost_of(indicator), 1.0, places=10)
        gradient = grads["mass"] + grads["stiff"]
        reference = _finite_difference(cost_of, indicator, nodes)
        for node, want in zip(nodes, reference):
            self.assertLess(abs(float(gradient[node]) - want), tol * abs(want))

    def test_it_matches_finite_differences_at_every_order(self):
        for order in (2, 4, 6):
            with self.subTest(order=order):
                sim = _sim((24, 26), space_order=order)
                self._check(sim, [(6, 9), (12, 12), (1, 10), (22, 11), (1, 1)])

    def test_it_matches_finite_differences_in_3D(self):
        sim = _sim((14, 13, 15), N=50, space_order=4)
        self._check(sim, [(6, 6, 7), (1, 6, 7), (12, 5, 8)])

    def test_it_matches_finite_differences_on_a_clamped_wall(self):
        sim = _sim((24, 26), space_order=4, boundary=Clamped)
        self._check(sim, [(6, 9), (1, 10), (1, 1)])

    def test_a_ghost_node_carries_no_gradient(self):
        sim = _sim((24, 26))
        source, sensors, indicator, objective = _problem(sim)
        _, grads, _, _ = sensitivity(sim, source, indicator, sensors, objective)
        gradient = grads["mass"] + grads["stiff"]
        for ghost in ((0, 12), (sim.Nx[0] - 1, 12), (12, 0), (12, sim.Nx[1] - 1)):
            self.assertEqual(float(gradient[ghost]), 0.0)

    def test_the_source_gradient_matches_finite_differences(self):
        sim = _sim((24, 26), space_order=4)
        source, sensors, indicator, objective = _problem(sim)

        def cost_of(src):
            return objective(simulate(sim, src, indicator, sensors=sensors)[1])[0]

        _, gradient, _, _ = source_sensitivity(
            sim, source, indicator, sensors, objective
        )
        h = 1e-6
        for t_index, column in ((20, 0), (40, 1), (10, 2)):
            up, dn = source.signal.copy(), source.signal.copy()
            up[t_index, column] += h
            dn[t_index, column] -= h
            want = (
                cost_of(Source(source.position, up))
                - cost_of(Source(source.position, dn))
            ) / (2.0 * h)
            got = float(gradient[t_index, column])
            self.assertLess(abs(got - want), 1e-4 * abs(want))


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class VariantTest(unittest.TestCase):
    def _reference(self, sim):
        """The solver arguments in call order, plus the exact cost and gradient."""
        source, sensors, indicator, objective = _problem(sim)
        args = (source, indicator, sensors, objective)
        cost, grads, _, _ = sensitivity(sim, *args)
        return args, cost, grads["mass"] + grads["stiff"]

    def test_reconstruction_is_the_exact_transpose(self):
        for Nx, order in (((24, 26), 4), ((14, 13, 15), 2)):
            with self.subTest(ndim=len(Nx), order=order):
                sim = _sim(Nx, N=50, space_order=order)
                args, cost, reference = self._reference(sim)
                got, grads, _, info = reconstruction_sensitivity(sim, *args)
                total = grads["mass"] + grads["stiff"]
                self.assertEqual(got, cost)
                self.assertEqual(info["strip"], 0)
                self.assertLess(
                    float(cp.linalg.norm(total - reference)),
                    1e-10 * float(cp.linalg.norm(reference)),
                )

    def test_the_sponge_is_reconstructed_exactly_where_it_is_lossless(self):
        # the strip must honour reach = 2r - 1: the strain and divergence taps compound
        for order in (2, 4):
            with self.subTest(order=order):
                res = 26
                dx = (1.0 / (res - 3),) * 2
                Nx, width, _, _ = pad_for_sponge((res, res), dx, 0.15, faces=(0, 1))
                sim = _sim(Nx, N=70, space_order=order)
                sim = replace(
                    sim,
                    damping=sponge(
                        sim,
                        cp.ones(sim.Nx_padded, dtype=sim.dtype),
                        width,
                        0.08,
                        faces=(0, 1),
                    ),
                )
                args, cost, reference = self._reference(sim)
                got, grads, _, info = reconstruction_sensitivity(sim, *args)
                _, valid = reconstruction_nodes(sim)
                total = (grads["mass"] + grads["stiff"]) * valid
                self.assertEqual(got, cost)
                self.assertGreater(info["strip"], 0)
                self.assertLess(
                    float(cp.linalg.norm(total - reference * valid)),
                    1e-10 * float(cp.linalg.norm(reference * valid)),
                )

    def test_superposition_agrees_in_direction(self):
        sim = _sim((24, 26), N=60, space_order=4)
        args, cost, reference = self._reference(sim)
        _, grads, _, _ = superposition_sensitivity(sim, *args, scale=1e-4)
        total = grads["mass"] + grads["stiff"]
        relative = float(cp.linalg.norm(total - reference)) / float(
            cp.linalg.norm(reference)
        )
        cosine = float(cp.sum(total * reference)) / (
            float(cp.linalg.norm(total)) * float(cp.linalg.norm(reference))
        )
        self.assertLess(relative, 0.25)
        self.assertGreater(cosine, 0.98)


# ------------------------------- transducers and validation --------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class TransducerTest(unittest.TestCase):
    def test_a_directed_receiver_scatters_as_it_gathers(self):
        sim = _sim((26, 26), N=20)
        length = (26 - 3) * sim.dx[0]
        sensors = Sensors(
            sim,
            [[0.3 * length, 0.9 * length], [0.6 * length, 0.9 * length]],
            direction=[0.0, 1.0],
        )
        rng = np.random.default_rng(2)
        x = cp.asarray(
            rng.standard_normal((sim.N, sensors.nodes.shape[1])), dtype=sim.dtype
        )
        y = cp.asarray(rng.standard_normal((sim.N, sensors.count)), dtype=sim.dtype)
        lhs = float(cp.sum(y * sensors.traces(x)))
        rhs = float(cp.sum(x * sensors.scatter(y)))
        self.assertLess(abs(lhs - rhs), 1e-12 * abs(lhs))

    def test_a_source_on_its_own_point_lands_on_one_unknown(self):
        sim = _sim((26, 26), N=5)
        source = point_source(
            sim,
            [[10.5 * sim.dx[0], 12.0 * sim.dx[1]]],
            np.ones(5),
            direction=[1.0, 0.0],
        )
        weights = cp.asnumpy(source.signal[0]).reshape(-1, sim.ncomp)
        self.assertEqual(int((np.abs(weights) > 1e-12).sum()), 1)

    def test_a_vector_unknown_needs_a_direction(self):
        sim = _sim((14, 14), N=5)
        with self.assertRaises(ValueError):
            point_source(sim, [[0.2, 0.2]], np.zeros(5))

    def test_the_clamped_wall_never_moves(self):
        sim = _sim((22, 22), N=60, boundary=Clamped)
        source, _, _, _ = _problem(sim)
        field = simulate(sim, source, cp.ones(sim.Nx_padded, dtype=sim.dtype))
        # the tangential components carry the wall unknowns: u_x on y-walls and back
        walls = [field[0][:, 1], field[0][:, 20], field[1][1, :], field[1][20, :]]
        for wall in walls:
            self.assertEqual(float(cp.abs(wall).max()), 0.0)
        self.assertGreater(float(cp.abs(field).max()), 0.0)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class TimestepTest(unittest.TestCase):
    def test_the_measured_timestep_agrees_with_the_cheap_bound_when_uniform(self):
        for order in (2, 4):
            sim = _sim((64, 64), N=1, space_order=order)
            uniform = cp.ones(sim.Nx_padded, dtype=sim.dtype)
            # stable_dt only knows the speed and the spacing, so it stays conservative
            self.assertGreater(stable_timestep(sim, uniform) / sim.dt, 1.0)

    def test_a_contrast_still_costs_timestep_at_a_wide_stencil(self):
        """The wide stencil reaches past the harmonic shear mean into a void, so order 4
        still pays timestep, though far less than the cell gather did."""
        res = 96
        ratios = {}
        for order in (2, 4):
            sim = _sim((res, res), N=1, space_order=order)
            axes = cp.meshgrid(
                *[
                    (cp.arange(n, dtype=sim.dtype) - 1) * h
                    for n, h in zip(sim.Nx_padded, sim.dx)
                ],
                indexing="ij",
            )
            void = (axes[0] - 0.5) ** 2 + (axes[1] - 0.5) ** 2 < 0.1**2
            indicator = cp.where(void, 1e-4, 1.0).astype(sim.dtype)
            ratios[order] = stable_timestep(sim, indicator) / sim.dt
        self.assertGreater(ratios[2], 1.0)
        self.assertLess(ratios[4], 1.0)
        self.assertGreater(ratios[4], 0.3)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class ValidationTest(unittest.TestCase):
    def test_it_rejects_a_missing_material(self):
        with self.assertRaises(ValueError):
            ElasticWave((10, 10), (0.1, 0.1), 1, 0.01, (8, 8), density=1.0)

    def test_it_rejects_a_shear_speed_above_the_pressure_one(self):
        with self.assertRaises(ValueError):
            ElasticWave(
                (10, 10),
                (0.1, 0.1),
                1,
                0.01,
                (8, 8),
                density=1.0,
                wavespeed_p=1.0,
                wavespeed_s=2.0,
            )

    def test_it_rejects_plane_stress_in_3D(self):
        with self.assertRaises(ValueError):
            _sim((8, 8, 8), N=1, plane="stress")

    def test_it_rejects_a_pressure_boundary_condition(self):
        from cuwave.boundary import Neumann

        with self.assertRaises(ValueError):
            _sim((10, 10), N=1, boundary=Neumann)


if __name__ == "__main__":
    unittest.main()
