"""Contracts of the cell-assembled elastic equation that the scalar tests cannot reach.

Three of them carry the rest. In 1D elasticity *is* the scalar equation, so
`AnisotropicElasticWave` must reproduce `ScalarWave` to round-off, which pins the whole
vector pipeline against the already-trusted one. The cell assembly is `-B^T C B`, so the
operator must be exactly symmetric, which is what all three adjoint variants rest on. And
full quadrature is what buys that symmetry without an hourglass mode, so the checkerboard
must not be a null mode.
"""

import unittest

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.anisotropic import AnisotropicElasticWave, cell_stencil, corner_bits
    from cuwave.boundary import Clamped
    from cuwave.elastic import voigt
    from cuwave.scalar import ScalarWave
    from cuwave.sensitivity import (
        l2_misfit,
        reconstruction_sensitivity,
        sensitivity,
        superposition_sensitivity,
    )
    from cuwave.signals import ricker
    from cuwave.utils import Sensors, point_source
    from cuwave.wave import (
        compile_kernels,
        define_step_method,
        simulate,
        stable_dt,
        stable_timestep,
    )

CP, CS, RHO = 2.0, 1.0, 1.3
THREADS = {1: (128,), 2: (8, 8), 3: (4, 4, 8)}


# -------------------------------------- helpers --------------------------------------
def _sim(Nx, N=80, precision="float64", **kwargs):
    dx = tuple(1.0 / (n - 3) for n in Nx)
    return AnisotropicElasticWave(
        Nx,
        dx,
        N,
        0.9 * stable_dt(dx, CP, 2),
        THREADS[len(Nx)][::-1],
        precision=precision,
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
    """The mass weighted spatial operator, which the assembly makes symmetric."""
    mat = sim.build_materials(indicator)
    step = define_step_method(sim, compile_kernels(sim), mat)
    interior = (Ellipsis, *(slice(1, n - 1) for n in sim.Nx))
    mass = cp.where(mat["minv"] > 0, 1.0 / cp.maximum(mat["minv"], 1e-300), 0.0)

    def apply(x):
        out = cp.zeros_like(x)
        step(cp.zeros_like(x), x, out)
        weighted = (2.0 * x - out) * mass
        masked = cp.zeros_like(weighted)
        masked[interior] = weighted[interior]
        return masked

    return apply, interior


def _finite_difference(cost_of, indicator, nodes, h=1e-6):
    """Central difference of the cost at each of `nodes`."""
    out = []
    for node in nodes:
        plus, minus = indicator.copy(), indicator.copy()
        plus[node] += h
        minus[node] -= h
        out.append((cost_of(plus) - cost_of(minus)) / (2.0 * h))
    return np.array(out)


# -------------------------------- element and operator -------------------------------
class StencilTest(unittest.TestCase):
    def test_the_stencil_has_only_the_rigid_body_null_space(self):
        for ndim, dx in ((1, (0.7,)), (2, (0.7, 1.3)), (3, (0.7, 1.3, 0.9))):
            with self.subTest(ndim=ndim):
                K = cell_stencil(ndim, dx, voigt(ndim, 2.0, 1.0))
                ev = np.linalg.eigvalsh(K)
                null = int(np.sum(np.abs(ev) < 1e-9 * abs(ev).max()))
                self.assertEqual(null, {1: 1, 2: 3, 3: 6}[ndim])
                self.assertLess(np.linalg.norm(K - K.T), 1e-12 * np.linalg.norm(K))

    def test_a_rigid_translation_carries_no_force(self):
        K = cell_stencil(2, (0.7, 1.3), voigt(2, 2.0, 1.0))
        for component in range(2):
            shift = np.zeros(K.shape[0])
            shift[component::2] = 1.0
            self.assertLess(np.linalg.norm(K @ shift), 1e-10 * np.linalg.norm(K))

    def test_the_corner_order_matches_the_kernel(self):
        self.assertEqual(corner_bits(2), [(0, 0), (1, 0), (0, 1), (1, 1)])


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class OperatorTest(unittest.TestCase):
    def test_the_kernel_reproduces_the_continuum_divergence(self):
        """Independent of every convention in the module: a smooth field against analytic
        div(sigma), on deliberately unequal dx so an axis mix-up cannot hide."""
        lame, shear = 1.0, 0.6
        a, b, c, e = 2.0 * np.pi, 3.0 * np.pi, np.pi, 2.0 * np.pi
        speed_p = np.sqrt((lame + 2.0 * shear) / RHO)
        speed_s = np.sqrt(shear / RHO)
        errors = []
        for n in (32, 64):
            dx = (1.0 / n, 1.3 / n)
            sim = AnisotropicElasticWave(
                (n + 3, n + 3),
                dx,
                1,
                0.9 * stable_dt(dx, speed_p, 2),
                (8, 32),
                precision="float64",
                density=RHO,
                wavespeed_p=speed_p,
                wavespeed_s=speed_s,
            )
            axes = [(np.arange(m) - 1) * h for m, h in zip(sim.Nx_padded, dx)]
            X, Y = np.meshgrid(*axes, indexing="ij")
            u = np.stack([np.sin(a * X) * np.cos(b * Y), np.cos(c * X) * np.sin(e * Y)])
            grad_x = -(a * a) * np.sin(a * X) * np.cos(b * Y) - e * c * np.sin(
                c * X
            ) * np.cos(e * Y)
            grad_y = -a * b * np.cos(a * X) * np.sin(b * Y) - (e * e) * np.cos(
                c * X
            ) * np.sin(e * Y)
            lap_x = -(a * a + b * b) * np.sin(a * X) * np.cos(b * Y)
            lap_y = -(c * c + e * e) * np.cos(c * X) * np.sin(e * Y)
            want = np.stack(
                [
                    (lame + shear) * grad_x + shear * lap_x,
                    (lame + shear) * grad_y + shear * lap_y,
                ]
            )
            u1 = cp.asarray(u, dtype=sim.dtype)
            mat = sim.build_materials(cp.ones(sim.Nx_padded, dtype=sim.dtype))
            step = define_step_method(sim, compile_kernels(sim), mat)
            u0, u2 = cp.zeros_like(u1), cp.zeros_like(u1)
            step(u0, u1, u2)
            cp.cuda.Stream.null.synchronize()
            got = cp.asnumpy(u2 - 2.0 * u1) * RHO / sim.dt**2
            inner = (Ellipsis, slice(3, n - 1), slice(3, n - 1))
            errors.append(
                np.abs(got[inner] - want[inner]).max() / np.abs(want[inner]).max()
            )
        self.assertLess(errors[0], 5e-2)
        self.assertGreater(np.log2(errors[0] / errors[1]), 1.7)

    def test_the_operator_is_symmetric(self):
        rng = np.random.default_rng(4)
        for Nx in ((18, 20), (12, 11, 13)):
            with self.subTest(ndim=len(Nx)):
                sim = _sim(Nx, N=1)
                gamma = cp.asarray(0.4 + rng.random(sim.Nx_padded), dtype=sim.dtype)
                apply, interior = _operator(sim, gamma)
                a = cp.zeros((sim.ncomp, *sim.Nx_padded), dtype=sim.dtype)
                b = cp.zeros_like(a)
                a[interior] = cp.asarray(rng.standard_normal(a[interior].shape))
                b[interior] = cp.asarray(rng.standard_normal(b[interior].shape))
                lhs = float(cp.sum(b * apply(a)))
                rhs = float(cp.sum(a * apply(b)))
                self.assertLess(abs(lhs - rhs), 1e-12 * abs(lhs))

    def test_the_checkerboard_is_not_a_null_mode(self):
        sim = _sim((18, 20), N=1)
        gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
        apply, interior = _operator(sim, gamma)
        axes = cp.meshgrid(
            *[cp.arange(n, dtype=sim.dtype) for n in sim.Nx_padded], indexing="ij"
        )
        checker = (-1.0) ** (axes[0] + axes[1])
        field = cp.zeros((sim.ncomp, *sim.Nx_padded), dtype=sim.dtype)
        field[:] = checker
        masked = cp.zeros_like(field)
        masked[interior] = field[interior]
        # full quadrature is what removes the hourglass mode of a one point cell
        self.assertGreater(
            float(cp.linalg.norm(apply(masked))), 1e-3 * float(cp.linalg.norm(masked))
        )

    def test_the_plane_wave_speeds_are_the_lame_ones(self):
        for ndim, plane, cp_ref in ((2, "strain", CP), (3, "strain", CP)):
            with self.subTest(ndim=ndim, plane=plane):
                dx = (0.01,) * ndim
                lame = RHO * (cp_ref**2 - 2.0 * CS**2)
                K = cell_stencil(ndim, dx, voigt(ndim, lame, RHO * CS**2, plane))
                corners = corner_bits(ndim)
                k = 2.0 * np.pi / (200 * dx[0])
                A = np.zeros((ndim, ndim), dtype=complex)
                for c in range(len(corners)):
                    own = sum((1 - ((c >> d) & 1)) << d for d in range(ndim))
                    for far in range(len(corners)):
                        off = np.array(
                            [((c >> d) & 1) - 1 + ((far >> d) & 1) for d in range(ndim)]
                        )
                        phase = np.exp(1j * k * off[0] * dx[0])
                        A += (
                            phase
                            * K[
                                own * ndim : own * ndim + ndim,
                                far * ndim : far * ndim + ndim,
                            ]
                        )
                A = (A + A.conj().T) / 2.0
                speeds = (
                    np.sqrt(
                        np.abs(np.linalg.eigvalsh(A).real / (RHO * float(np.prod(dx))))
                    )
                    / k
                )
                exact = np.sort([CS] * (ndim - 1) + [cp_ref])
                for got, want in zip(np.sort(speeds), exact):
                    self.assertAlmostEqual(got / want, 1.0, places=3)

    def test_plane_stress_softens_the_pressure_speed(self):
        lame, shear = RHO * (CP**2 - 2.0 * CS**2), RHO * CS**2
        strain = voigt(2, lame, shear, "strain")
        stress = voigt(2, lame, shear, "stress")
        self.assertLess(stress[0, 0], strain[0, 0])
        self.assertAlmostEqual(float(stress[2, 2]), float(strain[2, 2]), places=12)

    def test_a_supplied_stiffness_matrix_reaches_the_stencil(self):
        # a stiffened C must stiffen the assembled operator, at unchanged speeds
        stiffened = 2.0 * voigt(2, RHO * (CP**2 - 2.0 * CS**2), RHO * CS**2)
        base, aniso = _sim((14, 14), N=1), _sim((14, 14), N=1, C=stiffened)
        ratio = float(cp.abs(aniso.stencil()).max() / cp.abs(base.stencil()).max())
        self.assertAlmostEqual(ratio, 2.0, places=10)


# ---------------------------------- against the scalar -------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class OneDimensionTest(unittest.TestCase):
    """1D elasticity is the scalar equation, so the two solvers must agree exactly."""

    def _pair(self, N=400, res=300):
        Nx, dx = (res,), (1.0 / (res - 3),)
        dt = 0.7 * stable_dt(dx, CP, 2)
        common = dict(precision="float64", space_order=2)
        scalar = ScalarWave(Nx, dx, N, dt, (128,), wavespeed=CP, density=RHO, **common)
        elastic = AnisotropicElasticWave(
            Nx,
            dx,
            N,
            dt,
            (128,),
            density=RHO,
            wavespeed_p=CP,
            wavespeed_s=0.4 * CP,
            **common,
        )
        return scalar, elastic

    def test_it_reproduces_the_scalar_solver(self):
        scalar, elastic = self._pair()
        t = np.arange(scalar.N) * scalar.dt
        signal = ricker(t, 1.0, 12.0)
        volume = float(np.prod(scalar.dx))
        nodes = scalar.Nx_padded[0]
        cases = {
            "uniform": np.full(nodes, 0.8),
            "smooth": 0.7 + 0.25 * np.sin(np.linspace(0.0, 7.0, nodes)),
            "void": np.where(abs(np.arange(nodes) - 200) < 15, 1e-3, 1.0),
        }
        for label, values in cases.items():
            with self.subTest(gamma=label):
                gamma = cp.asarray(values, dtype=scalar.dtype)
                # the elastic source is a force, the scalar one a force density
                a = simulate(
                    scalar, point_source(scalar, [[0.35]], signal), gamma.copy()
                )
                b = simulate(
                    elastic,
                    point_source(elastic, [[0.35]], signal * volume, direction=[1.0]),
                    gamma.copy(),
                )
                a, b = cp.asnumpy(a)[1:-1], cp.asnumpy(b)[1:-1]
                self.assertLess(np.abs(a - b).max(), 1e-10 * np.abs(a).max())


# ------------------------------------- sensitivity -----------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class GradientTest(unittest.TestCase):
    def _check(self, sim, nodes, tol=1e-5):
        source, sensors, indicator, objective = _problem(sim)

        def cost_of(field):
            return objective(simulate(sim, source, field, sensors=sensors)[1])[0]

        cost, grads, _, _ = sensitivity(sim, source, indicator, sensors, objective)
        self.assertAlmostEqual(cost / cost_of(indicator), 1.0, places=10)
        gradient = grads["mass"] + grads["stiff"]
        reference = _finite_difference(cost_of, indicator, nodes)
        for node, want in zip(nodes, reference):
            self.assertLess(abs(float(gradient[node]) - want), tol * abs(want))

    def test_it_matches_finite_differences_in_2D(self):
        sim = _sim((24, 26))
        self._check(sim, [(6, 9), (12, 12), (1, 10), (22, 11), (1, 1)])

    def test_it_matches_finite_differences_in_3D(self):
        sim = _sim((14, 13, 15), N=50)
        self._check(sim, [(6, 6, 7), (1, 6, 7), (12, 5, 8)])

    def test_a_ghost_node_carries_no_gradient(self):
        sim = _sim((24, 26))
        source, sensors, indicator, objective = _problem(sim)
        _, grads, _, _ = sensitivity(sim, source, indicator, sensors, objective)
        gradient = grads["mass"] + grads["stiff"]
        for ghost in ((0, 12), (sim.Nx[0] - 1, 12), (12, 0), (12, sim.Nx[1] - 1)):
            self.assertEqual(float(gradient[ghost]), 0.0)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class VariantTest(unittest.TestCase):
    def _reference(self, sim):
        """The solver arguments in call order, plus the exact cost and gradient."""
        source, sensors, indicator, objective = _problem(sim)
        args = (source, indicator, sensors, objective)
        cost, grads, _, _ = sensitivity(sim, *args)
        return args, cost, grads["mass"] + grads["stiff"]

    def test_reconstruction_is_the_exact_transpose(self):
        for Nx in ((24, 26), (14, 13, 15)):
            with self.subTest(ndim=len(Nx)):
                sim = _sim(Nx, N=50)
                args, cost, reference = self._reference(sim)
                got, grads, _, info = reconstruction_sensitivity(sim, *args)
                total = grads["mass"] + grads["stiff"]
                self.assertEqual(got, cost)
                self.assertEqual(info["strip"], 0)
                self.assertLess(
                    float(cp.linalg.norm(total - reference)),
                    1e-10 * float(cp.linalg.norm(reference)),
                )

    def test_superposition_agrees_in_direction(self):
        sim = _sim((24, 26), N=60)
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

    def test_the_clamped_wall_never_moves(self):
        sim = _sim((22, 22), N=60, boundary=Clamped)
        source, _, _, _ = _problem(sim)
        field = simulate(sim, source, cp.ones(sim.Nx_padded, dtype=sim.dtype))
        for wall in (1, sim.Nx[0] - 2):
            self.assertEqual(float(cp.abs(field[:, wall, :]).max()), 0.0)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class ValidationTest(unittest.TestCase):
    def test_it_rejects_a_missing_material(self):
        with self.assertRaises(ValueError):
            AnisotropicElasticWave((10, 10), (0.1, 0.1), 1, 0.01, (8, 8), density=1.0)

    def test_it_rejects_a_wide_stencil(self):
        # the cell gather costs (2r)**(2 ndim): a wide stencil belongs to ElasticWave
        with self.assertRaises(ValueError):
            _sim((10, 10), N=1, space_order=4)

    def test_it_rejects_an_asymmetric_stiffness_matrix(self):
        C = voigt(2, 1.0, 0.5)
        C[0, 1] += 1.0
        with self.assertRaises(ValueError):
            _sim((10, 10), N=1, C=C)

    def test_it_rejects_plane_stress_in_3D(self):
        with self.assertRaises(ValueError):
            _sim((8, 8, 8), N=1, plane="stress")

    def test_the_measured_timestep_agrees_with_the_cheap_bound_when_uniform(self):
        sim = _sim((64, 64), N=1)
        uniform = cp.ones(sim.Nx_padded, dtype=sim.dtype)
        # stable_dt only knows the speed and the spacing, so it stays conservative
        self.assertGreater(stable_timestep(sim, uniform) / sim.dt, 1.0)


if __name__ == "__main__":
    unittest.main()
