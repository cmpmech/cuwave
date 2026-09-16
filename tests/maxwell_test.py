"""Contracts of the two out-of-plane Maxwell reductions.

Three of them carry the rest. The reductions are `PressureWave` with its two coefficient
fields exchanged, so `ElectricWave` must reproduce the acoustic solver under the change of
variables, unknown for unknown, which is what says the physics is a renaming and not a
rewrite. The permittivity has to reach the kernel as a permittivity, so a slab has to
transmit what Fabry-Perot says it transmits, which is the one check that ties the material
units, the spectral objective and the source calibration together. And the design enters
`ElectricWave` through the inertia but `MagneticWave` through the stiffness, so only the
first keeps an exact adjoint above `space_order` 2: the wide-stencil gradient is pinned for
one and deliberately not claimed for the other.
"""

import math
import unittest

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.boundary import Conductor, Magnetic, Traction
    from cuwave.geometry import box
    from cuwave.maxwell import (
        DielectricWave,
        ElectricWave,
        MagneticWave,
        interpolate,
    )
    from cuwave.scalar import AcousticWave
    from cuwave.sensitivity import (
        reconstruction_sensitivity,
        sensitivity,
        superposition_sensitivity,
    )
    from cuwave.signals import ricker
    from cuwave.stencils import staggered_weights
    from cuwave.utils import Sensors, intensity, point_source, response_gradient
    from cuwave.wave import (
        compile_kernels,
        define_step_method,
        grid_coords,
        simulate,
        stable_dt,
    )

INDEX = 3.48
EPS1, EPS2, MU = 1.0, INDEX**2, 1.0
THREADS = {1: (128,), 2: (8, 8)}
VECTOR_THREADS = {2: (8, 8), 3: (4, 4, 8)}


if HAS_CUDA:

    class MagnetodielectricWave(DielectricWave):
        """Designs the permeability alongside the permittivity, to reach `USE_MAGNETIC`."""

        magnetic = True
        permeability2: float = 2.0

        def parametrization(self, indicator):
            """Both materials interpolate, so both densities carry a design dependence."""
            return (
                interpolate(indicator, self.permittivity1, self.permittivity2),
                1.0 / interpolate(indicator, self.permeability, self.permeability2),
            )

        def parametrization_jacobian(self, indicator):
            """The permittivity is affine in gamma; the inverse permeability is not."""
            contrast = self.permeability2 - self.permeability
            mu = interpolate(indicator, self.permeability, self.permeability2)
            return self.permittivity2 - self.permittivity1, -contrast / mu**2


# -------------------------------------- helpers --------------------------------------
def _sim(cls, Nx, N=260, length=1.0, space_order=2, **kwargs):
    dx = tuple(length / (n - 3) for n in Nx)
    return cls(
        Nx,
        dx,
        N,
        0.7 * stable_dt(dx, 1.0 / math.sqrt(EPS1 * MU), space_order),
        THREADS[len(Nx)],
        precision="float64",
        space_order=space_order,
        permittivity1=EPS1,
        permittivity2=EPS2,
        permeability=MU,
        **kwargs,
    )


def _problem(sim, seed=0):
    """A shot, a small receiver set and a rough design, shared by the gradient checks."""
    rng = np.random.default_rng(seed)
    t = np.arange(sim.N) * sim.dt
    source = point_source(sim, [(0.3, 0.5)], ricker(t, 1.0, 8.0))
    sensors = cp.asarray([[30, 30, 32], [24, 26, 25]], dtype=cp.int32)
    indicator = cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype)
    return source, sensors, indicator


def _finite_difference(cost_of, indicator, nodes, h=1e-7):
    """Central difference of the cost at each of `nodes`."""
    out = []
    for node in nodes:
        plus, minus = indicator.copy(), indicator.copy()
        plus[node] += h
        minus[node] -= h
        out.append((cost_of(plus) - cost_of(minus)) / (2.0 * h))
    return out


def _slab_transmission(space_order, points_per_wavelength, frequencies):
    """Measured and analytic power transmission of a quarter-wave silicon slab in 1D."""
    thickness = 0.25 / INDEX
    length, x_src, x_slab, x_probe = 20.0, 8.0, 10.0, 12.0
    dx = (1.0 / (INDEX * points_per_wavelength),)
    Nx = (int(length / dx[0]) + 3,)
    dt = 0.95 * stable_dt(dx, 1.0, space_order)
    # both wall echoes reach the probe at t = 20, so the record stops just short of it
    N = math.ceil(19.0 / dt)

    sim = ElectricWave(
        Nx,
        dx,
        N,
        dt,
        (128,),
        precision="float64",
        space_order=space_order,
        permittivity1=EPS1,
        permittivity2=EPS2,
        permeability=MU,
    )
    coords = grid_coords(Nx, dx, dtype=sim.dtype)
    vacuum = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    slab = box(coords, (x_slab + 0.5 * thickness,), (thickness,)).astype(sim.dtype)

    t = np.arange(N) * dt
    source = point_source(sim, [(x_src,)], ricker(t, 1.0, 1.2))
    probe = cp.asarray([[round(x_probe / dx[0]) + 1]], dtype=cp.int32)
    reference = simulate(sim, source, vacuum, sensors=probe)[1].get()[:, 0]
    through = simulate(sim, source, slab, sensors=probe)[1].get()[:, 0]

    basis = np.exp(-2j * np.pi * np.outer(frequencies, t))
    measured = np.abs((basis @ through) / (basis @ reference)) ** 2
    beta = 2.0 * np.pi * INDEX * float(slab.sum()) * dx[0] * np.asarray(frequencies)
    contrast = (EPS1 - EPS2) ** 2 / (4.0 * EPS1 * EPS2)
    return measured, 1.0 / (1.0 + contrast * np.sin(beta) ** 2)


# --------------------------------- the two reductions --------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class ReductionTest(unittest.TestCase):
    def test_it_is_the_acoustic_solver_under_the_change_of_variables(self):
        em = _sim(ElectricWave, (64, 64))
        ac = AcousticWave(
            em.Nx,
            em.dx,
            em.N,
            em.dt,
            THREADS[2],
            precision="float64",
            space_order=em.space_order,
            rho1=MU,
            rho2=MU,
            kappa1=1.0 / EPS1,
            kappa2=1.0 / EPS2,
        )
        gamma = cp.asarray(
            np.random.default_rng(1).random(em.Nx_padded), dtype=em.dtype
        )
        t = np.arange(em.N) * em.dt
        signal = ricker(t, 1.0, 8.0)
        one = simulate(em, point_source(em, [(0.4, 0.5)], signal), gamma)
        other = simulate(ac, point_source(ac, [(0.4, 0.5)], signal), gamma)
        self.assertEqual(float(cp.max(cp.abs(one - other))), 0.0)

    def test_a_plane_wave_travels_at_the_speed_the_index_sets(self):
        for eps in (EPS1, EPS2):
            index = math.sqrt(eps * MU)
            dx = (1.0 / 1997,)
            dt = 0.4 * stable_dt(dx, 1.0 / index, 2)
            sim = ElectricWave(
                (2000,),
                dx,
                int(0.6 * index / dt),
                dt,
                (128,),
                precision="float64",
                permittivity1=eps,
                permittivity2=eps,
                permeability=MU,
            )
            t = np.arange(sim.N) * dt
            source = point_source(sim, [(0.2,)], ricker(t, 1.0, 30.0))
            probes = cp.asarray(
                [[round(0.5 / dx[0]) + 1, round(0.7 / dx[0]) + 1]], dtype=cp.int32
            )
            field = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
            peaks = cp.argmax(
                cp.abs(simulate(sim, source, field, sensors=probes)[1]), axis=0
            ).get()
            speed = 0.2 / ((peaks[1] - peaks[0]) * dt)
            self.assertAlmostEqual(speed * index, 1.0, delta=0.01)

    def test_the_light_speed_is_the_faster_of_the_two_phases(self):
        sim = _sim(ElectricWave, (32, 32))
        self.assertAlmostEqual(sim.light_speed, 1.0 / math.sqrt(EPS1 * MU))

    def test_it_rejects_three_dimensions(self):
        with self.assertRaises(ValueError):
            ElectricWave(
                (16, 16, 16),
                (0.1,) * 3,
                10,
                0.01,
                (4, 4, 8),
                permittivity1=EPS1,
                permittivity2=EPS2,
            )

    def test_it_rejects_a_missing_permittivity(self):
        with self.assertRaises(ValueError):
            ElectricWave((32, 32), (0.1, 0.1), 10, 0.01, (8, 8), permittivity1=1.0)


# ------------------------------------ transmission -----------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class TransmissionTest(unittest.TestCase):
    frequencies = (0.5, 1.0, 1.5, 2.0)

    def test_the_slab_transmission_matches_the_fabry_perot_formula(self):
        measured, analytic = _slab_transmission(2, 20, self.frequencies)
        for f, m, a in zip(self.frequencies, measured, analytic):
            self.assertAlmostEqual(m / a, 1.0, delta=0.05, msg=f"at frequency {f}")

    def test_refining_the_grid_sharpens_it(self):
        # the interface limits this, not the stencil, so it converges at first order
        coarse, analytic = _slab_transmission(2, 20, self.frequencies)
        fine, _ = _slab_transmission(2, 80, self.frequencies)

        def worst(measured):
            return float(np.max(np.abs(measured - analytic) / analytic))

        self.assertLess(worst(fine), 0.25 * worst(coarse))


# ------------------------------------- gradients -------------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class GradientTest(unittest.TestCase):
    frequencies = (6.0, 10.0)

    def _check(self, cls, space_order, adjoint=sensitivity, tolerance=1e-4):
        sim = _sim(cls, (48, 48), space_order=space_order)
        source, sensors, indicator = _problem(sim)
        objective = intensity(sim, self.frequencies)
        rng = np.random.default_rng(5)
        nodes = [
            (int(rng.integers(12, 34)), int(rng.integers(12, 34))) for _ in range(4)
        ]

        def cost_of(field):
            return response_gradient(
                sim, source, field, sensors, objective, adjoint=adjoint
            )[0]

        _, gradient = response_gradient(
            sim, source, indicator, sensors, objective, adjoint=adjoint
        )
        for node, reference in zip(
            nodes, _finite_difference(cost_of, indicator, nodes)
        ):
            self.assertAlmostEqual(
                float(gradient[node]) / reference, 1.0, delta=tolerance, msg=f"{node}"
            )

    def test_an_inertia_design_matches_finite_differences(self):
        self._check(ElectricWave, 2)

    def test_a_stiffness_design_matches_finite_differences(self):
        self._check(MagneticWave, 2)

    def test_an_inertia_design_survives_a_wide_stencil(self):
        # dJ/deps carries no stencil, so the wide-order transpose error never reaches it
        self._check(ElectricWave, 4, tolerance=1e-3)
        self._check(ElectricWave, 6, tolerance=1e-3)

    def test_the_reverse_march_reproduces_the_stored_gradient(self):
        self._check(ElectricWave, 2, adjoint=reconstruction_sensitivity)
        self._check(MagneticWave, 2, adjoint=reconstruction_sensitivity)


# ---------------------------------- the vector case ----------------------------------
def _vector(Nx, N=150, space_order=2, cls=None, **kwargs):
    dx = tuple(1.0 / (n - 3) for n in Nx)
    return (cls or DielectricWave)(
        Nx,
        dx,
        N,
        0.6 * stable_dt(dx, 1.0 / math.sqrt(EPS1 * MU), space_order),
        VECTOR_THREADS[len(Nx)],
        precision="float64",
        space_order=space_order,
        permittivity1=EPS1,
        permittivity2=4.0,
        permeability=MU,
        **kwargs,
    )


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


def _staggered_gradient(sim, phi):
    """The half-point gradient of `phi` at the field points, at the simulation's own order."""
    radius = sim.space_order // 2
    coefficients = staggered_weights(radius)
    out = cp.zeros(sim.field_shape, dtype=sim.dtype)
    inner = tuple(slice(radius, n - radius) for n in sim.Nx)
    for c in range(sim.ncomp):
        for j, weight in enumerate(coefficients, start=1):

            def window(offset, c=c):
                return tuple(
                    slice(radius + offset, n - radius + offset)
                    if d == c
                    else slice(radius, n - radius)
                    for d, n in enumerate(sim.Nx)
                )

            out[c][inner] += (
                weight / sim.dx[c] * (phi[window(j)] - phi[window(-(j - 1))])
            )
    return out


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class VectorMaxwellTest(unittest.TestCase):
    grids = ((22, 26), (14, 16, 18))
    orders = (2, 4, 6)

    def test_the_operator_is_symmetric_at_every_order_and_boundary(self):
        rng = np.random.default_rng(0)
        for Nx in self.grids:
            walls = (
                ("conductor", None),
                ("magnetic", Magnetic),
                ("mixed", (Conductor, Magnetic) + ((Magnetic,) * (len(Nx) - 2))),
            )
            for order in self.orders:
                for name, boundary in walls:
                    with self.subTest(grid=Nx, order=order, boundary=name):
                        sim = _vector(Nx, N=40, space_order=order, boundary=boundary)
                        gamma = cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype)
                        apply, mask = _operator(sim, gamma)
                        x = mask(
                            cp.asarray(
                                rng.standard_normal(sim.field_shape), dtype=sim.dtype
                            )
                        )
                        y = mask(
                            cp.asarray(
                                rng.standard_normal(sim.field_shape), dtype=sim.dtype
                            )
                        )
                        ax, ay = apply(x), apply(y)
                        scale = float(cp.linalg.norm(ax)) * float(cp.linalg.norm(y))
                        gap = abs(float(cp.sum(y * ax)) - float(cp.sum(x * ay)))
                        self.assertLess(gap / scale, 1e-12)

    def test_a_discrete_gradient_is_a_null_mode(self):
        # catches a curl sign or index error that symmetry alone cannot see
        rng = np.random.default_rng(1)
        for Nx in ((30, 34), (18, 20, 22)):
            for order in self.orders:
                with self.subTest(grid=Nx, order=order):
                    sim = _vector(Nx, N=40, space_order=order, boundary=Magnetic)
                    gamma = cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype)
                    apply, _ = _operator(sim, gamma)
                    phi = cp.asarray(
                        rng.standard_normal(sim.Nx_padded), dtype=sim.dtype
                    )
                    gradient = _staggered_gradient(sim, phi)
                    # away from the walls, where no stencil is graded down
                    margin = sim.space_order + 2
                    inner = (slice(None), *(slice(margin, n - margin) for n in sim.Nx))
                    residual = float(cp.linalg.norm(apply(gradient)[inner]))
                    self.assertLess(
                        residual, 1e-10 * float(cp.linalg.norm(gradient[inner]))
                    )

    def test_a_conductor_wall_holds_the_tangential_field_at_zero(self):
        sim = _vector((24, 26), N=80)
        gamma = cp.asarray(
            np.random.default_rng(2).random(sim.Nx_padded), dtype=sim.dtype
        )
        t = np.arange(sim.N) * sim.dt
        source = point_source(sim, [(0.5, 0.5)], ricker(t, 1.0, 8.0), [0.0, 1.0])
        field = simulate(sim, source, gamma)
        for d in range(sim.ndim):
            for index in (1, sim.Nx[d] - 2):
                wall = [slice(None)] * sim.ndim
                wall[d] = index
                for c in range(sim.ncomp):
                    if c != d:
                        self.assertEqual(float(cp.max(cp.abs(field[(c, *wall)]))), 0.0)

    def test_it_rejects_one_dimension(self):
        with self.assertRaises(ValueError):
            DielectricWave(
                (32,), (0.1,), 10, 0.01, (128,), permittivity1=1.0, permittivity2=4.0
            )

    def test_it_rejects_an_elastic_face(self):
        with self.assertRaises(ValueError):
            _vector((16, 16), boundary=Traction)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class VectorGradientTest(unittest.TestCase):
    frequencies = (6.0, 10.0)

    def _problem(self, sim, seed=4):
        rng = np.random.default_rng(seed)
        t = np.arange(sim.N) * sim.dt
        direction = [0.0, 1.0] + [0.0] * (sim.ndim - 2)
        source = point_source(sim, [[0.35] * sim.ndim], ricker(t, 1.0, 8.0), direction)
        receivers = Sensors(sim, [[0.6] * sim.ndim], direction)
        objective = receivers.objective(intensity(sim, self.frequencies))
        indicator = cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype)
        return source, receivers, objective, indicator

    def _check(self, Nx, space_order, adjoint=sensitivity, cls=None, tolerance=1e-4):
        sim = _vector(Nx, space_order=space_order, cls=cls)
        source, receivers, objective, indicator = self._problem(sim)
        rng = np.random.default_rng(7)

        def cost_of(field):
            return response_gradient(
                sim, source, field, receivers.nodes, objective, adjoint=adjoint
            )[0]

        _, gradient = response_gradient(
            sim, source, indicator, receivers.nodes, objective, adjoint=adjoint
        )
        nodes = [tuple(int(rng.integers(6, n - 6)) for n in Nx) for _ in range(3)]
        for node, reference in zip(
            nodes, _finite_difference(cost_of, indicator, nodes)
        ):
            self.assertAlmostEqual(
                float(gradient[node]) / reference, 1.0, delta=tolerance, msg=f"{node}"
            )

    def test_the_gradient_matches_finite_differences(self):
        for Nx, order in (((26, 28), 2), ((26, 28), 4), ((16, 18, 20), 2)):
            with self.subTest(grid=Nx, order=order):
                self._check(Nx, order)

    def test_the_reverse_march_reproduces_the_stored_gradient(self):
        self._check((26, 28), 2, adjoint=reconstruction_sensitivity)

    def test_a_designed_permeability_matches_finite_differences(self):
        # the USE_MAGNETIC path, which no shipped parametrization exercises
        self._check((26, 28), 2, cls=MagnetodielectricWave)
        self._check((16, 18, 20), 2, cls=MagnetodielectricWave)

    def test_superposition_agrees_in_direction(self):
        sim = _vector((24, 26), N=60)
        source, receivers, objective, indicator = self._problem(sim)
        _, reference, _, _ = sensitivity(
            sim, source, indicator, receivers.nodes, objective
        )
        _, grads, _, _ = superposition_sensitivity(
            sim, source, indicator, receivers.nodes, objective, scale=1e3
        )
        truth = reference["mass"]
        total = grads["mass"]
        cosine = float(cp.sum(total * truth)) / (
            float(cp.linalg.norm(total)) * float(cp.linalg.norm(truth))
        )
        self.assertGreater(cosine, 0.98)


if __name__ == "__main__":
    unittest.main()
