"""Contracts of the two out-of-plane Maxwell reductions.

Three of them carry the rest. The reductions are `PressureWave` with its two
coefficient fields exchanged, so `ElectricWave` must reproduce the acoustic solver
under the change of variables, unknown for unknown, which is what says the physics is a
renaming and not a rewrite. The permittivity has to reach the kernel as a permittivity,
so a slab has to transmit what Fabry-Perot says it transmits, which is the one check
that ties the material units, the spectral objective and the source calibration
together. And the design enters `ElectricWave` through the inertia but `MagneticWave`
through the stiffness, so only the first keeps its gradient in the interior above
`space_order` 2 (the wide wall closure spoils it only next to a wall): the wide-stencil
gradient is pinned for one, away from the walls, and deliberately not for the other.

Independently of all of that, fdtdx (Mahlau et al. 2026, an E/H Yee leapfrog) is rerun
here from its stored probe traces: eliminating H from Yee's scheme gives exactly this
curl-curl recursion, so wherever both codes place the permittivity alike the traces
agree at round-off, and where they do not (the 3D component mean, the TE cell mean)
the gap is pinned to the placement alone. `spectral_sensitivity` is pinned to the stored
gradient on a damped domain, where both of its runs decay.

A 2D float32 vector run on a small grid advances several steps a launch
(`define_blocked_step`), the curl, the update, the excitation and the record inside the
tile. It claims the curl and update pair bit for bit, so it is compared with
`array_equal` over orders, faces, damping, tiles and sources on both components. Past L2
the pair becomes one launch (`_define_fused`) that recomputes the curl of a ring around
each tile, pinned to the pair bit for bit in 2D and 3D, a designed permeability included.
"""

import itertools
import math
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave import maxwell
    from cuwave.boundary import (
        Conductor,
        Dirichlet,
        Magnetic,
        Neumann,
        Pml,
        Traction,
        sponge,
    )
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
        spectral_sensitivity,
        superposition_sensitivity,
    )
    from cuwave.signals import ricker
    from cuwave.stencils import staggered_weights
    from cuwave.utils import Sensors, intensity, point_source, response_gradient
    from cuwave.wave import (
        Source,
        compile_kernels,
        flatten_indices,
        grid_coords,
        simulate,
        stable_dt,
    )

REFERENCE = Path(__file__).parent / "reference"
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

    def _check(self, cls, space_order, adjoint=None, tolerance=1e-4):
        adjoint = sensitivity if adjoint is None else adjoint
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
        # dJ/deps carries no stencil, so the closure error reaches it only near a wall
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
    step = sim.define_step(compile_kernels(sim), mat)
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
    coefficients = staggered_weights(radius, sim.stencil)
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

    def test_a_long_run_decays_behind_either_wall(self):
        for wall in (Conductor, Magnetic):
            with self.subTest(wall=wall):
                sim = _vector((60, 60), N=16000, space_order=4, boundary=wall)
                sim = replace(sim, pml=Pml(12, 1.0, frequency=1.0, faces=(0, 1, 3)))
                t = np.arange(sim.N) * sim.dt
                # a second difference has both moments zero, so no charge is left behind
                pulse = np.diff(np.exp(-((t - 1.0) * 6.0) ** 2), 2, prepend=0, append=0)
                signal = cp.asarray(pulse[:, None] / sim.dt**2, dtype=sim.dtype)
                source = Source(cp.array([[1], [30], [30]], dtype=cp.int32), signal)
                rng = np.random.default_rng(1)
                indicator = cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype)
                sensors = cp.array([[0, 1], [30, 14], [14, 30]], dtype=cp.int32)
                um = cp.abs(simulate(sim, source, indicator, sensors)[1])
                windows = [float(w.max()) for w in cp.split(um, 8)]
                self.assertLess(windows[-1], 1e-4 * windows[0])
                self.assertLess(windows[-1], windows[-2])

    def test_an_unshifted_layer_is_refused(self):
        with self.assertRaises(ValueError):
            _vector((40, 40), pml=Pml(6, 1.0))


def _blocked_runs(
    order, Nx, N, boundary, damped=False, blocks=None, waves=(1000, 1000)
):
    """The (field, record) of a 2D float32 run forced blocked, then of the plain pair."""
    dx = tuple(1.0 / (n - 3) for n in Nx)
    dt = 0.9 * stable_dt(dx, 1.0, order)
    sim = DielectricWave(
        Nx,
        dx,
        N,
        dt,
        (8, 32),
        space_order=order,
        permittivity1=1.0,
        permittivity2=4.0,
        permeability=1.0,
        boundary=boundary,
    )
    rng = np.random.default_rng(order)
    indicator = cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype)
    if damped:
        sim = replace(sim, damping=sponge(sim, indicator, 12, 0.3))
    n = sim.Nx
    position = [[0, 1, 1], [0, n[0] // 2, 7], [1, 3, n[1] - 3], [1, n[0] - 3, 2]]
    position = cp.array(np.array(position).T, dtype=cp.int32)
    signal = cp.asarray(rng.standard_normal((N, position.shape[1])), dtype=sim.dtype)
    near = [[c, a, 5] for c in (0, 1) for a in range(1, 12)]
    sensors = near + [[0, 1, 1], [1, n[0] - 2, n[1] - 2], [0, n[0] - 3, n[1] - 2]]
    sensors = cp.array(np.array(sensors).T, dtype=cp.int32)
    source = Source(position, signal)
    patches = {"BLOCKED_WAVES": waves}
    patches |= {"BLOCKED": blocks} if blocks is not None else {}
    with mock.patch.multiple(maxwell, **patches):
        mat = sim.build_materials(indicator)
        assert sim.define_blocked_step(mat, source, sensors) is not None
        blocked = simulate(sim, source, indicator, sensors=sensors)
    with mock.patch.object(DielectricWave, "define_blocked_step", lambda *a, **k: None):
        plain = simulate(sim, source, indicator, sensors=sensors)
    return blocked, plain


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class BlockedTest(unittest.TestCase):
    def _assert_bitwise(self, blocked, plain):
        self.assertGreater(float(cp.abs(plain[0]).max()), 0.0)
        for a, b in zip(blocked, plain):
            self.assertTrue(bool(cp.array_equal(a, b)))

    def test_a_blocked_run_is_the_plain_pair_bit_for_bit(self):
        mixed = ((Conductor, Magnetic), (Magnetic, Conductor))
        for order in (2, 4):
            for boundary in (Conductor, Magnetic, mixed):
                for damped in (False, True):
                    with self.subTest(order=order, b=boundary, damped=damped):
                        runs = _blocked_runs(order, (150, 127), 41, boundary, damped)
                        self._assert_bitwise(*runs)

    def test_the_large_tier_is_the_plain_pair_bit_for_bit(self):
        mixed = ((Conductor, Magnetic), (Magnetic, Conductor))
        cases = itertools.product((2, 4), (Conductor, Magnetic, mixed), (False, True))
        for order, boundary, damped in cases:
            with self.subTest(order=order, b=boundary, damped=damped):
                # no small tier, and the large one inside L2 on any GPU
                with mock.patch.object(maxwell, "LARGE_SMS", (0, 0)):
                    runs = _blocked_runs(
                        order, (150, 127), 41, boundary, damped, waves=(0, 0)
                    )
                self._assert_bitwise(*runs)

    def test_every_tile_and_step_count(self):
        blocks = (
            ((32, 32), 32, 6),
            ((32, 64), 16, 6),
            ((64, 64), 16, 12),
            ((32, 128), 8, 6),
            ((32, 64), 8, 3),
            ((32, 32), 32, 2),
        )
        for order, block in [(o, b) for o in (2, 4) for b in blocks]:
            if min(block[0]) - 2 * block[2] * (order - 1) < 8:
                continue
            with self.subTest(order=order, block=block):
                runs = _blocked_runs(
                    order, (147, 115), 30, Conductor, False, ((block,),) * 2
                )
                self._assert_bitwise(*runs)

    def test_what_the_pair_keeps(self):
        dx = (0.01, 0.01)
        common = dict(permittivity1=1.0, permittivity2=4.0, permeability=1.0)
        sim = DielectricWave((150, 127), dx, 6, 1e-3, (8, 32), **common)
        signal = cp.zeros((6, 2), dtype=sim.dtype)
        once = Source(cp.array([[0, 0], [50, 60], [40, 40]], dtype=cp.int32), signal)
        twice = Source(cp.array([[0, 0], [50, 50], [40, 40]], dtype=cp.int32), signal)
        indicator = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
        with mock.patch.object(maxwell, "BLOCKED_WAVES", (1000, 1000)):
            mat = sim.build_materials(indicator)
            self.assertIsNotNone(sim.define_blocked_step(mat, once, None))
            self.assertIsNone(sim.define_blocked_step(mat, twice, None))
            wide = replace(sim, precision="float64")
            mat = wide.build_materials(indicator.astype(wide.dtype))
            self.assertIsNone(wide.define_blocked_step(mat, once, None))


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class FusedTest(unittest.TestCase):
    def test_the_fused_step_is_the_pair_bit_for_bit(self):
        rng = np.random.default_rng(5)
        mixed = ((Conductor, Magnetic), (Magnetic, Conductor))
        mixed3 = (*mixed, (Conductor, Conductor))
        cases = (
            ((37, 70), 2, Conductor, "float32", False, DielectricWave),
            ((37, 70), 4, mixed, "float32", True, DielectricWave),
            ((37, 70), 8, Magnetic, "float32", False, MagnetodielectricWave),
            ((37, 70), 4, mixed, "float64", False, DielectricWave),
            ((13, 21, 40), 2, mixed3, "float32", True, DielectricWave),
            ((13, 21, 40), 4, Conductor, "float32", False, MagnetodielectricWave),
            ((13, 21, 40), 2, Magnetic, "float64", False, DielectricWave),
        )
        for Nx, order, boundary, precision, damped, cls in cases:
            with self.subTest(Nx=Nx, order=order, cls=cls.__name__, damped=damped):
                dx = tuple(1.0 / (n - 3) for n in Nx)
                threads = VECTOR_THREADS[len(Nx)]
                common = dict(permittivity1=EPS1, permittivity2=4.0, permeability=MU)
                sim = cls(
                    Nx,
                    dx,
                    1,
                    1e-3,
                    threads,
                    precision=precision,
                    space_order=order,
                    boundary=boundary,
                    **common,
                )
                gamma = cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype)
                if damped:
                    sim = replace(sim, damping=sponge(sim, gamma, 3, 0.3))
                mat = sim.build_materials(gamma)
                # a grid this small takes the pair
                pair = sim._define_pair(compile_kernels(sim), mat, adjoint=False)
                tile = maxwell.FUSED_TILE[sim.ndim - 2]
                ring = math.prod(t + order for t in tile)
                fused = sim._define_fused(
                    mat, tile, sim.npairs * ring * sim.dtype().itemsize
                )
                shape = (sim.ncomp, *sim.Nx_padded)
                u0, u1 = (
                    cp.asarray(rng.standard_normal(shape), dtype=sim.dtype)
                    for _ in range(2)
                )
                a = pair(u0, u1, cp.zeros_like(u0))
                b = fused(u0, u1, cp.zeros_like(u0))
                self.assertGreater(float(cp.abs(a).max()), 0.0)
                self.assertTrue(bool(cp.array_equal(a, b)))


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

    def _check(self, Nx, space_order, adjoint=None, cls=None, tolerance=1e-4, **kw):
        adjoint = sensitivity if adjoint is None else adjoint
        sim = _vector(Nx, space_order=space_order, cls=cls, **kw)
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

    def test_a_designed_permeability_matches_finite_differences_through_a_pml(self):
        layer = Pml(4, 1.0 / math.sqrt(EPS1 * MU), frequency=1.0)
        self._check((26, 28), 2, cls=MagnetodielectricWave, pml=layer)
        self._check((16, 18, 20), 2, cls=MagnetodielectricWave, pml=layer)

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


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class SpectralTest(unittest.TestCase):
    frequencies = (6.0, 10.0)

    def _check(self, sim, beta=0.02):
        rng = np.random.default_rng(3)
        t = np.arange(sim.N) * sim.dt
        indicator = cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype)
        minv = sim.inverse_inertia(indicator)
        sim = replace(sim, damping=(2.0 * beta / sim.dt / minv).astype(sim.dtype))
        direction = None if sim.ncomp == 1 else [0.0, 1.0] + [0.0] * (sim.ndim - 2)
        source = point_source(sim, [[0.35] * sim.ndim], ricker(t, 1.0, 8.0), direction)
        receivers = Sensors(sim, [[0.6] * sim.ndim], direction)
        objective = receivers.objective(intensity(sim, self.frequencies))
        args = (sim, source, indicator, receivers.nodes, objective)
        _, reference, _, _ = sensitivity(*args)
        _, grads, _, info = spectral_sensitivity(*args, self.frequencies)
        d_mass, d_stiff = sim.parametrization_jacobian(indicator)
        truth = d_mass * reference["mass"] + d_stiff * reference["stiff"]
        found = d_mass * grads["mass"] + d_stiff * grads["stiff"]
        rel = float(cp.linalg.norm(found - truth) / cp.linalg.norm(truth))
        self.assertLess(info["decay"], 1e-5)
        self.assertLess(rel, 1e-5, f"difference {rel:.3e}")

    def test_the_reductions_match_the_stored_gradient(self):
        for cls in (ElectricWave, MagneticWave):
            for order in (2, 4):
                with self.subTest(cls=cls.__name__, order=order):
                    self._check(_sim(cls, (28, 30), N=1400, space_order=order))

    def test_the_vector_field_matches_the_stored_gradient(self):
        for Nx, cls in (
            ((26, 28), None),
            ((16, 18, 20), None),
            ((26, 28), MagnetodielectricWave),
        ):
            with self.subTest(grid=Nx, cls=cls):
                self._check(_vector(Nx, N=1400, cls=cls))


def _fdtdx(name):
    """Rerun fdtdx reference `name` here: (our probe traces, fdtdx's), same grid and dt."""
    ref = np.load(REFERENCE / f"fdtdx_{name}.npz")
    shape = tuple(int(n) for n in ref["shape"])
    ndim, S = len(shape), float(ref["courant"])
    cls, wall, row = {
        "tm2D": (ElectricWave, Dirichlet, []),
        "te2D": (MagneticWave, Neumann, []),
        "vector3D": (DielectricWave, Conductor, [2]),
    }[name.split("_")[0]]
    # fdtdx cell i is node i + 1, so the ghost ring wraps the conductor faces
    sim = cls(
        tuple(n + 2 for n in shape),
        (1.0,) * ndim,
        len(ref["amplitude"]),
        S,
        (8, 32) if ndim == 2 else (2, 8, 32),
        precision="float64",
        permittivity1=1.0,
        permittivity2=float(ref["permittivity"]),
        boundary=((wall, wall),) * ndim,
    )
    indicator = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    if ref["block_low"].size:
        low, size = ref["block_low"] + 1, ref["block_size"]
        indicator[tuple(slice(lo, lo + n) for lo, n in zip(low, size))] = 1.0
    q = -S * ref["amplitude"]
    source = np.array(ref["source"]) + 1
    if cls is MagneticWave:
        below = source - np.eye(ndim, dtype=int)[1]
        nodes, loads = [source, below], [-S * q, S * q]
    else:
        nodes, loads = [source], [np.diff(q, prepend=0.0)]
    position = cp.asarray(
        [row * len(nodes)] * len(row) + np.array(nodes).T.tolist(), dtype=cp.int32
    )
    mat = sim.build_materials(indicator)
    weight = cp.asnumpy(sim.excitation_weights(mat, flatten_indices(sim, position)))
    signal = cp.asarray(np.stack(loads, axis=1) / weight, dtype=sim.dtype)
    probes = ref["probes"].T + 1
    probes = cp.asarray(
        [row * probes.shape[1]] * len(row) + probes.tolist(), dtype=cp.int32
    )
    traces = simulate(sim, Source(position, signal), indicator, sensors=probes)[1]
    return cp.asnumpy(traces), ref["traces"]


def _gap(name):
    ours, theirs = _fdtdx(name)
    return float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs))


@unittest.skipUnless(
    HAS_CUDA and REFERENCE.is_dir(), "needs a CUDA device and tests/reference"
)
class FdtdxReferenceTest(unittest.TestCase):
    def test_a_shared_placement_reproduces_fdtdx_at_round_off(self):
        for name in ("tm2D_vacuum", "tm2D_block", "te2D_vacuum", "vector3D_vacuum"):
            with self.subTest(case=name):
                self.assertLess(_gap(name), 1e-12)

    def test_a_staircased_permittivity_closes_the_3D_gap(self):
        # fdtdx gives a component the voxel below it, which is no averaging at all
        self.assertLess(_gap("vector3D_block"), 1e-3)
        with mock.patch("cuwave.maxwell.point_harmonic", lambda sim, field, c: field):
            self.assertLess(_gap("vector3D_block"), 1e-12)

    def test_the_te_placement_gap_closes_under_refinement(self):
        coarse, fine = _gap("te2D_block"), _gap("te2D_block_fine")
        self.assertLess(coarse, 5e-3)
        self.assertLess(fine, 0.5 * coarse)


if __name__ == "__main__":
    unittest.main()
