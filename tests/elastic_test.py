"""Contracts of the staggered elastic equation that the scalar tests cannot reach.

Three of them carry the rest. The staggered strains put every stress on a natural
point, so the operator must converge at its full order while the cost stays flat in the
radius. The operator is `-B^T C B` with matching taps in both kernels, so it must be
exactly symmetric at every order and boundary mix, which is what all adjoint variants
rest on. And the staggering shifts every component off the nodes, so sources, sensors
and the reconstruction strip must all honour the half-node offsets. The shear terms
couple diagonal neighbours, so the strip of a layer lining both axes needs its corners.

`FusedTest` pins the one-launch 2D step a grid past L2 takes to the stress and update
pair bit for bit: it recomputes the stresses of a ring around each tile, so a ring too
narrow or a wall term read off the wrong tile would show at once.

`AdjointGradientTest` pins the adjoint pair that carries the gradients to the step and
the gradient kernel launched apart, on both tiers: the step and the stiffness densities
bit for bit, the inertia density to its sum by parts in time.

`BlockedTest` pins the 2D step a small grid advances several steps a launch to the pair,
the excitation and the record launched one by one, bit for bit, over orders, walls,
damping, step counts and a grid whose last tile core holds one node.

`PmlTest` pins the two conditions the elastic pml is only stable under, both refused
when broken: the CFS shift, without which the cross terms grow a static mode, and a
clamped wall behind the layer, since a traction wall there holds a growing surface wave.
Held to both, a long run through a heterogeneous model with a free surface crossing the
layer has to decay rather than grow.

A point force inside a uniformly damped body has to deliver the momentum of a damped
particle: the internal forces cancel in the sum over the body, so the summed displacement,
over the undamped run's, follows the particle's recursion to round-off. That pins the
source's share of the damped divisor `1 + beta` to the inertia the kernel divides by.

A `Relaxation` is pinned where it can fail independently: the fit against its own
complex modulus at the frequencies the step sees, a plane wave between two receivers of
a 1D line against the target Q and the reference phase speed, and its adjoint as the
exact transpose (a linear objective, so the forward impulse responses are the oracle at
round-off) and against finite differences for the material gradient.
"""

import itertools
import math
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave import elastic
    from cuwave.boundary import Clamped, Pml, Traction, pad_for_sponge, sponge
    from cuwave.elastic import ElasticWave, Relaxation, zener
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
        simulate,
        stable_dt,
        stable_timestep,
    )

CP, CS, RHO = 2.0, 1.0, 1.3
THREADS = {1: (128,), 2: (8, 8), 3: (4, 4, 8)}
RELAXATION = Relaxation(20.0, (2.0, 20.0), 6.0) if HAS_CUDA else None


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
    step = sim.define_step(compile_kernels(sim), mat)
    every = cp.arange(sim.ncomp * sim.comp_stride, dtype=cp.int32)
    minv = sim.point_inverse_inertia(mat["gamma"], every).reshape(sim.field_shape)
    mass = cp.where(minv > 0, 1.0 / cp.maximum(minv, 1e-300), 0.0)
    slices = [
        (c, *(slice(1, n - 1 - (d == c)) for d, n in enumerate(sim.Nx)))
        for c in range(sim.ncomp)
    ]

    def mask(x):
        out = cp.zeros_like(x)
        for sl in slices:
            out[sl] = x[sl]
        return out * (minv > 0)

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


def _particle(beta, signal):
    """Displacement of a unit particle after the damped three-term recursion under `signal`."""
    x = np.zeros(len(signal) + 2)
    for t, s in enumerate(signal):
        x[t + 2] = (2.0 * x[t + 1] - (1.0 - beta) * x[t] + s) / (1.0 + beta)
    return x[-1]


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
        stencil="taylor",  # formal order holds for the Taylor rows only
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
    step = sim.define_step(compile_kernels(sim), mat)
    u0, u2 = cp.zeros_like(u1), cp.zeros_like(u1)
    step(u0, u1, u2)
    cp.cuda.Stream.null.synchronize()
    got = cp.asnumpy(u2 - 2.0 * u1) * RHO / sim.dt**2
    margin = order + 1
    inner = (Ellipsis, slice(margin, n - margin), slice(margin, n - margin))
    return np.abs(got[inner] - want[inner]).max() / np.abs(want[inner]).max()


# -------------------------------- stencil and operator -------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
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
            step = sim.define_step(compile_kernels(sim), mat)
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

    def test_it_matches_finite_differences_with_an_optimized_stencil(self):
        sim = _sim((24, 26), space_order=6, stencil="optimized")
        self._check(sim, [(6, 9), (12, 12), (1, 10), (22, 11)])

    def test_it_matches_finite_differences_in_3D(self):
        sim = _sim((14, 13, 15), N=50, space_order=4)
        self._check(sim, [(6, 6, 7), (1, 6, 7), (12, 5, 8)])

    def test_it_matches_finite_differences_on_a_clamped_wall(self):
        sim = _sim((24, 26), space_order=4, boundary=Clamped)
        self._check(sim, [(6, 9), (1, 10), (1, 1)])

    def test_it_matches_finite_differences_under_a_relaxation(self):
        for boundary in (Traction, Clamped):
            for order in (2, 4):
                with self.subTest(boundary=boundary, order=order):
                    sim = _sim(
                        (24, 26),
                        space_order=order,
                        boundary=boundary,
                        relaxation=RELAXATION,
                    )
                    self._check(sim, [(6, 9), (12, 12), (1, 10), (22, 11), (1, 1)])
        sim = _sim((14, 13, 15), N=50, space_order=4, relaxation=RELAXATION)
        self._check(sim, [(6, 6, 7), (1, 6, 7), (12, 5, 8)])

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
        for order, faces in ((2, (0, 1)), (4, (0, 1)), (2, None), (4, None)):
            with self.subTest(order=order, faces=faces):
                res = 26
                dx = (1.0 / (res - 3),) * 2
                Nx, width, _, _ = pad_for_sponge((res, res), dx, 0.15, faces=faces)
                sim = _sim(Nx, N=70, space_order=order)
                sim = replace(
                    sim,
                    damping=sponge(
                        sim,
                        cp.ones(sim.Nx_padded, dtype=sim.dtype),
                        width,
                        0.08,
                        faces=faces,
                    ),
                )
                args, cost, reference = self._reference(sim)
                got, grads, _, info = reconstruction_sensitivity(sim, *args)
                _, valid = reconstruction_nodes(sim, args[1])
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
class FusedTest(unittest.TestCase):
    def test_the_fused_step_is_the_pair_bit_for_bit(self):
        rng = np.random.default_rng(5)
        mixed = ((Traction, Clamped), (Clamped, Traction))
        mixed3 = (*mixed, (Traction, Traction))
        cases = (
            ((37, 70), 2, None, "float32", False),
            ((37, 70), 4, Clamped, "float32", False),
            ((37, 70), 4, mixed, "float32", True),
            ((37, 70), 8, None, "float32", False),
            ((37, 70), 4, mixed, "float64", False),
            ((13, 21, 40), 2, Clamped, "float32", False),
            ((13, 21, 40), 4, mixed3, "float32", True),
            ((13, 21, 40), 2, None, "float64", False),
        )
        for Nx, order, boundary, precision, damped in cases:
            with self.subTest(Nx=Nx, order=order, b=boundary, damped=damped):
                sim = _sim(Nx, 1, precision, order, boundary=boundary)
                gamma = cp.asarray(0.4 + rng.random(sim.Nx_padded), dtype=sim.dtype)
                if damped:
                    sim = replace(sim, damping=sponge(sim, gamma, 3, 0.3))
                mat = sim.build_materials(gamma)
                # a grid this small takes the pair
                pair = sim._define_pair(compile_kernels(sim), mat, adjoint=False)
                fused = sim._define_fused(mat)
                shape = (sim.ncomp, *sim.Nx_padded)
                u0, u1 = (
                    cp.asarray(rng.standard_normal(shape), dtype=sim.dtype)
                    for _ in range(2)
                )
                a = pair(u0, u1, cp.zeros_like(u0))
                b = fused(u0, u1, cp.zeros_like(u0))
                self.assertGreater(float(cp.abs(a).max()), 0.0)
                self.assertTrue(bool(cp.array_equal(a, b)))


def _assert_fused_adjoint(test, sim, gamma, rng, fused):
    """The adjoint step carrying the gradients against the step, the gradient kernel and the sum by parts."""
    mat = sim.build_materials(gamma)
    kernels = compile_kernels(sim, sim.sensitivity_path)
    l0, l1, u1 = (
        cp.asarray(rng.standard_normal(sim.field_shape), dtype=sim.dtype)
        for _ in range(3)
    )
    step = sim.define_adjoint_step(compile_kernels(sim), kernels, mat)
    expected = step(l0, l1, cp.zeros_like(l0))
    separate, grads = sim.gradient_fields(mat), sim.gradient_fields(mat)
    zero = cp.zeros_like(u1)
    sim.define_gradient(kernels, mat, separate)(zero, u1, zero, l1)
    got = l0.copy()
    with mock.patch.object(type(sim), "_fuses_pair", lambda self: fused):
        sim.define_adjoint_gradient(kernels, mat, grads)(got, l1, u1)
    # the points the step writes, each component short of its staggered ghost
    points = cp.zeros((sim.ndim, *sim.Nx_padded), dtype=cp.bool_)
    for c in range(sim.ndim):
        points[c][tuple(slice(1, n - 1 - (d == c)) for d, n in enumerate(sim.Nx))] = 1
    points = points.reshape(sim.field_shape)
    test.assertTrue(bool(cp.array_equal(got[points], expected[points])))
    for name in set(separate) - {"mass"}:
        test.assertTrue(bool(cp.array_equal(grads[name], separate[name])), name)
    mass = -sim.adjoint_mass_factor() * u1 * (expected - 2 * l1 + l0) * points
    error = float(cp.abs(grads["mass"] - mass).max() / cp.abs(mass).max())
    test.assertLess(error, 1e-13)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class AdjointGradientTest(unittest.TestCase):
    def test_the_fused_adjoint_step_is_the_step_and_the_gradient_kernel(self):
        rng = np.random.default_rng(6)
        mixed = ((Traction, Clamped), (Clamped, Traction))
        cases = (
            ((40,), 4, Clamped, None, False),
            ((37, 70), 2, None, None, False),
            ((37, 70), 4, mixed, "damped", True),
            ((37, 70), 4, Clamped, "pml", False),
            ((13, 21, 40), 4, (*mixed, (Traction, Traction)), "damped", False),
            ((13, 21, 40), 2, Clamped, None, True),
        )
        for Nx, order, boundary, extra, tiers in cases:
            for fused in (False, True) if tiers else (False,):
                with self.subTest(Nx=Nx, order=order, extra=extra, fused=fused):
                    sim = _sim(Nx, 1, "float64", order, boundary=boundary)
                    gamma = cp.asarray(0.4 + rng.random(sim.Nx_padded), dtype=sim.dtype)
                    if extra == "damped":
                        sim = replace(sim, damping=sponge(sim, gamma, 3, 0.3))
                    if extra == "pml":
                        sim = replace(sim, pml=Pml(4, CP, frequency=1.0))
                    _assert_fused_adjoint(self, sim, gamma, rng, fused)


def _blocked_runs(sim, seed, blocks=None, waves=(99, 99)):
    """The (field, record) of a run blocked as `blocks` and `waves` patch, then of the pair's run."""
    rng = np.random.default_rng(seed)
    gamma = cp.asarray(0.25 + 0.5 * rng.random(sim.Nx_padded), dtype=sim.dtype)
    coords = [[0.3, 0.4], [0.05, 0.9], [0.9, 0.06]]
    signal = rng.standard_normal((sim.N, 3))
    source = point_source(sim, coords, signal, direction=[0.6, 0.8])
    n = sim.Nx
    rows = [[0, 1, 1], [1, 2, 3], [0, n[0] - 3, 5], [1, 9, n[1] - 2], [0, 21, 22]]
    rows += [[1, 22, 23]]
    sensors = cp.array(np.array(rows).T, dtype=cp.int32)
    candidates = elastic.BLOCKED if blocks is None else blocks
    with mock.patch.multiple(elastic, BLOCKED=candidates, BLOCKED_WAVES=waves):
        mat = sim.build_materials(gamma)
        assert sim.define_blocked_step(mat, source, sensors) is not None
        blocked = simulate(sim, source, gamma, sensors=sensors)
    with mock.patch.object(ElasticWave, "define_blocked_step", lambda *a, **k: None):
        plain = simulate(sim, source, gamma, sensors=sensors)
    return blocked, plain


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class BlockedTest(unittest.TestCase):
    def _assert_bitwise(self, blocked, plain):
        self.assertGreater(float(cp.abs(plain[0]).max()), 0.0)
        for a, b in zip(blocked, plain):
            self.assertTrue(bool(cp.array_equal(a, b)))

    def test_a_blocked_run_is_the_pair_bit_for_bit(self):
        mixed = ((Traction, Clamped), (Clamped, Traction))
        for order in (2, 4):
            for boundary in (None, Clamped, mixed):
                for damped in (False, True):
                    for N in (3, 40):
                        with self.subTest(order=order, b=boundary, damped=damped, N=N):
                            sim = _sim((70, 57), N, "float32", order, boundary=boundary)
                            if damped:
                                rho = cp.ones(sim.Nx_padded, dtype=sim.dtype)
                                sim = replace(sim, damping=sponge(sim, rho, 9, 0.3))
                            self._assert_bitwise(*_blocked_runs(sim, order))

    def test_the_large_tier_is_the_pair_bit_for_bit(self):
        mixed = ((Traction, Clamped), (Clamped, Traction))
        for boundary, damped in itertools.product((None, mixed), (False, True)):
            with self.subTest(b=boundary, damped=damped):
                sim = _sim((70, 57), 40, "float32", 2, boundary=boundary)
                if damped:
                    rho = cp.ones(sim.Nx_padded, dtype=sim.dtype)
                    sim = replace(sim, damping=sponge(sim, rho, 9, 0.3))
                # no small tier, and the large one inside L2 on any GPU
                with mock.patch.object(elastic, "LARGE_SMS", 0):
                    self._assert_bitwise(*_blocked_runs(sim, 2, waves=(0, 0)))

    def test_every_tile_and_step_count(self):
        blocks = (
            ((32, 32), 32, 6),
            ((32, 32), 32, 3),
            ((32, 64), 8, 4),
            ((32, 32), 16, 2),
            ((32, 64), 16, 4),
            ((48, 64), 16, 6),
        )
        for order, block in [(o, b) for o in (2, 4) for b in blocks]:
            if min(block[0]) - 2 * block[2] * (order - 1) < 8:
                continue
            with self.subTest(order=order, block=block):
                sim = _sim((70, 57), 30, "float32", order, boundary=Clamped)
                self._assert_bitwise(*_blocked_runs(sim, 1, ((block,),) * 2))

    def test_a_last_core_of_one_node(self):
        # cores of 24: each axis's last core holds the node beside the ghost alone
        sim = _sim((99, 51), 30, "float32", 2)
        self._assert_bitwise(*_blocked_runs(sim, 2, ((((32, 32), 32, 4),),) * 2))


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class PmlTest(unittest.TestCase):
    def test_a_long_run_decays_with_a_free_surface_crossing_the_layer(self):
        dx = (1.0 / 100,) * 2
        sim = ElasticWave(
            (110, 100),
            dx,
            16000,
            0.9 * stable_dt(dx, 1.0, 2),
            (4, 32),
            density=1.0,
            wavespeed_p=1.0,
            wavespeed_s=0.6,
            boundary=((Clamped, Clamped), (Traction, Clamped)),
            pml=Pml(14, 1.0, frequency=8.0, faces=(0, 1, 3)),
        )
        t = np.arange(sim.N) * sim.dt
        signal = cp.asarray(ricker(t, 1.0, 8.0)[:, None], dtype=sim.dtype)
        source = Source(cp.array([[1], [40], [50]], dtype=cp.int32), signal)
        rng = np.random.default_rng(1)
        indicator = cp.asarray(0.5 + rng.random(sim.Nx_padded), dtype=sim.dtype)
        sensors = cp.array([[0, 1], [50, 50], [50, 50]], dtype=cp.int32)
        um = cp.abs(simulate(sim, source, indicator, sensors)[1])
        windows = [float(w.max()) for w in cp.split(um, 8)]
        self.assertLess(windows[-1], 1e-3 * windows[0])
        self.assertLess(windows[-1], windows[-2])

    def test_an_unshifted_layer_is_refused(self):
        with self.assertRaises(ValueError):
            _sim((40, 40), boundary=Clamped, pml=Pml(6, CP))

    def test_a_traction_wall_behind_the_layer_is_refused(self):
        with self.assertRaises(ValueError):
            _sim((40, 40), pml=Pml(6, CP, frequency=2.0))
        layer = Pml(6, CP, frequency=2.0, faces=(0, 1, 3))
        free_top = ((Clamped, Clamped), (Traction, Clamped))
        _sim((40, 40), boundary=free_top, pml=layer)


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
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

    def test_a_damped_body_sums_to_a_damped_particle(self):
        alpha = 8.0
        for order in (2, 4):
            with self.subTest(order=order):
                sim = _sim((64, 64), N=40, space_order=order)
                gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
                damping = 2.0 * alpha / sim.inverse_inertia(gamma)
                damped = replace(sim, damping=damping.astype(sim.dtype))
                signal = np.ones(sim.N)
                source = point_source(sim, [[0.5, 0.5]], signal, direction=[1.0, 0.0])
                undamped, found = (  # the x components, along the force
                    float(cp.sum(simulate(s, source, gamma)[0])) for s in (sim, damped)
                )
                expected = _particle(alpha * sim.dt, signal) / _particle(0.0, signal)
                self.assertLess(abs(found / undamped - expected), 1e-9 * expected)


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


# ------------------------------------- relaxation ------------------------------------
def _line(quality, frequency):
    """The direct pulse at x = 1.6 and 2.6 of a relaxed 1D line fired at x = 0.6, and the line."""
    dx = 2.5e-3
    dt = 0.5 * stable_dt((dx,), 1.0, 4)
    sim = ElasticWave(
        (round(3.6 / dx) + 3,),
        (dx,),
        math.ceil(2.6 / dt),
        dt,
        THREADS[1],
        precision="float64",
        space_order=4,
        density=1.0,
        wavespeed_p=1.0,
        wavespeed_s=0.5,
        relaxation=Relaxation(quality, (2.0, 20.0), 6.0),
    )
    t = np.arange(sim.N) * dt
    source = point_source(sim, [[0.6]], ricker(t, 1.0, frequency))
    sensors = cp.asarray([[round(1.6 / dx) + 1, round(2.6 / dx) + 1]], dtype=cp.int32)
    gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    traces = simulate(sim, source, gamma, sensors=sensors)[1].get()
    # flat over the pulse, zero before the wall echoes arrive 1.2 behind it
    arrival = np.array([1.0, 2.0]) + 1.0 / frequency
    ramp = np.clip((0.35 - np.abs(t[:, None] - arrival)) / 0.1, 0.0, 1.0)
    return traces * (0.5 - 0.5 * np.cos(np.pi * ramp)), sim


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class RelaxationTest(unittest.TestCase):
    def test_a_relaxed_line_holds_its_quality_and_its_reference_speed(self):
        traces, sim = _line(20.0, 12.0)
        f = np.fft.rfftfreq(8 * sim.N, sim.dt)
        spectra = np.fft.rfft(traces, 8 * sim.N, axis=0)
        ratio = spectra[:, 1] / spectra[:, 0]
        # the wavenumber over the unit between the receivers, its modulus ~ 1 / k**2
        k = -np.unwrap(np.angle(ratio)) + 1j * np.log(np.abs(ratio))
        modulus = 1.0 / k[(f >= 3.0) & (f <= 18.0)] ** 2
        quality = np.abs(modulus.real / modulus.imag)
        self.assertLess(np.abs(quality / 20.0 - 1.0).max(), 0.02)
        at = np.argmin(np.abs(f - 6.0))
        self.assertLess(abs(2.0 * np.pi * f[at] / k[at].real - 1.0), 1e-3)

    def test_the_fit_holds_quality_where_the_step_sees_the_band(self):
        dt = 1e-3
        for quality, band, count in ((10.0, (1.0, 10.0), 3), (130.0, (1.0, 100.0), 5)):
            with self.subTest(quality=quality, band=band):
                rates, weights, _ = zener(Relaxation(quality, band, 1.0, count), dt)
                f = np.geomspace(*band, 400)
                omega = 2.0 / dt * np.tan(np.pi * f * dt)[:, None]
                modulus = 1.0 - np.sum(weights * rates / (rates + 1j * omega), axis=1)
                found = modulus.real / modulus.imag
                self.assertLess(np.abs(found / quality - 1.0).max(), 0.01)

    def test_the_relaxed_adjoint_is_the_exact_transpose(self):
        for sim, damped in (
            (_sim((24, 26), space_order=4, relaxation=RELAXATION), True),
            (_sim((14, 13, 15), N=50, space_order=4, relaxation=RELAXATION), False),
        ):
            with self.subTest(ndim=sim.ndim):
                source, sensors, indicator, _ = _problem(sim)
                if damped:
                    sim = replace(sim, damping=sponge(sim, indicator, 4, 0.2))
                rng = np.random.default_rng(1)
                weights = rng.standard_normal((sim.N, sensors.shape[1]))
                weights = cp.asarray(weights, dtype=sim.dtype)

                def objective(um):
                    return float(cp.sum(weights * um)), weights

                _, gradient, _, _ = source_sensitivity(
                    sim, source, indicator, sensors, objective
                )
                for t_index, column in ((5, 0), (20, 1), (41, 2)):
                    unit = cp.zeros_like(source.signal)
                    unit[t_index, column] = 1.0
                    record = simulate(
                        sim, Source(source.position, unit), indicator, sensors=sensors
                    )[1]
                    want = objective(record)[0]
                    got = float(gradient[t_index, column])
                    self.assertLess(abs(got - want), 1e-12 * abs(want))

    def test_the_variants_that_reverse_or_rebuild_the_field_refuse_it(self):
        sim = _sim((24, 26), relaxation=RELAXATION)
        source, sensors, indicator, objective = _problem(sim)
        for variant in (reconstruction_sensitivity, superposition_sensitivity):
            with self.subTest(variant=variant.__name__):
                with self.assertRaises(NotImplementedError):
                    variant(sim, source, indicator, sensors, objective)

    def test_it_refuses_a_pml_and_a_band_past_nyquist(self):
        with self.assertRaises(ValueError):
            _sim(
                (30, 30),
                N=1,
                boundary=Clamped,
                pml=Pml(6, 2.0, frequency=2.0),
                relaxation=RELAXATION,
            )
        sim = _sim((30, 30), N=1)
        with self.assertRaises(ValueError):
            replace(sim, relaxation=Relaxation(20.0, (2.0, 1.0 / sim.dt), 6.0))


if __name__ == "__main__":
    unittest.main()
