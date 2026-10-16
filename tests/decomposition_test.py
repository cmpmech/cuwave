"""Contracts of splitting axis 0 of one grid over several devices.

A subdomain runs the unmodified kernels over its rows and a halo, so the only thing
that can go wrong is the halo: too thin, exchanged too late, or missing a piece of the
state (a pml memory, a stored history row). Every one of those shows as a difference on
the owned rows, so the split run is pinned bitwise against the whole one, with a source
and sensors inside the halos and on a cut. Several entries naming the same GPU make
the subdomains threads on one card, which is all a test machine is sure to have.
"""

import functools
import unittest
from dataclasses import replace

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.boundary import Clamped, Pml
    from cuwave.elastic import ElasticWave
    from cuwave.maxwell import DielectricWave
    from cuwave.scalar import AcousticWave, ScalarWave
    from cuwave.sensitivity import (
        reconstruction_sensitivity,
        sensitivity,
        spectral_sensitivity,
        superposition_sensitivity,
    )
    from cuwave.signals import ricker
    from cuwave.utils import Sensors, measure, misfit_gradient, point_source
    from cuwave.wave import simulate, stable_dt

DEVICES = (0, 0, 0)


# -------------------------------------- helpers --------------------------------------
def _scalar(order=4, absorber=None, Nx=(220, 130)):
    dx = (0.01,) * len(Nx)
    dt = 0.8 * stable_dt(dx, 1.0, order)
    pml = Pml(10, 1.0) if absorber == "pml" else None
    sim = ScalarWave(
        Nx,
        dx,
        200,
        dt,
        (32, 8) if len(Nx) == 2 else (32, 4, 4),
        precision="float64",
        space_order=order,
        pml=pml,
        wavespeed=1.0,
        density=1.0,
    )
    if absorber == "sponge":
        damping = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
        damping[:20], damping[-20:], damping[..., -20:] = 30.0, 30.0, 30.0
        sim = replace(sim, damping=damping)
    return sim, None


def _elastic(order=4, absorber=None, Nx=(220, 130)):
    dx = (0.01,) * len(Nx)
    dt = 0.8 * stable_dt(dx, 2.0, order)
    pml = Pml(10, 2.0, frequency=5.0) if absorber == "pml" else None
    sim = ElasticWave(
        Nx,
        dx,
        200,
        dt,
        (32, 8) if len(Nx) == 2 else (32, 4, 4),
        precision="float64",
        space_order=order,
        boundary=Clamped if pml else None,
        pml=pml,
        density=1.0,
        wavespeed_p=2.0,
        wavespeed_s=1.0,
    )
    return sim, [0.0] * (len(Nx) - 1) + [1.0]


def _experiment(sim, direction):
    """A two-point source and sensors on the cuts of three subdomains and in their halos."""
    tail = [0.3] * (sim.ndim - 1)
    signal = ricker(np.arange(sim.N) * sim.dt, 1.0, 10.0)
    source = point_source(sim, [[1.0, *tail], [0.3, *tail]], signal, direction)
    rows = [0.25, 0.72, 0.99, 1.1, 1.46, 1.95]
    sensors = Sensors(sim, [[x, *tail] for x in rows], direction)
    gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    gamma[80:120, 40:60] = 0.7
    return source, sensors, gamma


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class ForwardTest(unittest.TestCase):
    def _check(self, sim, direction):
        source, sensors, gamma = _experiment(sim, direction)
        field, record = simulate(sim, source, gamma, sensors=sensors.nodes)
        split = replace(sim, devices=DEVICES)
        split_field, split_record = simulate(
            split, source, gamma, sensors=sensors.nodes
        )
        np.testing.assert_array_equal(split_field.get(), field.get())
        np.testing.assert_array_equal(split_record.get(), record.get())

    def test_the_split_scalar_field_is_the_whole_one(self):
        for order in (2, 4, 8):
            for absorber in (None, "sponge", "pml"):
                with self.subTest(order=order, absorber=absorber):
                    self._check(*_scalar(order, absorber, Nx=(300, 130)))

    def test_the_split_elastic_field_is_the_whole_one(self):
        for absorber in (None, "pml"):
            with self.subTest(absorber=absorber):
                self._check(*_elastic(4, absorber, Nx=(300, 130)))

    def test_the_split_3D_field_is_the_whole_one(self):
        for make in (_scalar, _elastic):
            with self.subTest(equation=make.__name__):
                sim, direction = make(4, "pml", Nx=(220, 36, 40))
                self._check(replace(sim, N=100), direction)


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class GradientTest(unittest.TestCase):
    def _check(self, sim, direction, adjoint, rtol=0.0):
        source, sensors, gamma = _experiment(sim, direction)
        observed = measure(sim, [source], cp.ones_like(gamma), sensors)
        args = ([source], gamma, sensors, observed)
        cost, gradient = misfit_gradient(sim, *args, adjoint=adjoint)
        split = replace(sim, devices=DEVICES)
        split_cost, split_gradient = misfit_gradient(split, *args, adjoint=adjoint)
        self.assertEqual(split_cost, cost)
        scale = float(cp.abs(gradient).max())
        np.testing.assert_allclose(
            split_gradient.get(), gradient.get(), rtol=0.0, atol=rtol * scale
        )

    def test_the_split_gradients_are_the_whole_ones(self):
        for make in (_scalar, _elastic):
            for absorber in (None, "sponge", "pml"):
                for adjoint in (sensitivity, reconstruction_sensitivity):
                    with self.subTest(make=make.__name__, absorber=absorber):
                        self._check(*make(4, absorber), adjoint)

    def test_the_acoustic_and_maxwell_gradients_split_alike(self):
        common = dict(precision="float64", space_order=4)
        dt = 0.5 * stable_dt((0.01, 0.01), 2.0, 4)
        sims = (
            AcousticWave(
                (220, 130),
                (0.01, 0.01),
                200,
                dt,
                (32, 8),
                rho1=1.0,
                rho2=3.0,
                kappa1=1.0,
                kappa2=5.0,
                pml=Pml(10, 1.0),
                **common,
            ),
            DielectricWave(
                (220, 130),
                (0.01, 0.01),
                200,
                dt,
                (32, 8),
                permittivity1=1.0,
                permittivity2=4.0,
                pml=Pml(10, 1.0, frequency=1.0),
                **common,
            ),
        )
        for sim, direction in zip(sims, (None, [0.0, 1.0])):
            for adjoint in (sensitivity, reconstruction_sensitivity):
                with self.subTest(sim=type(sim).__name__, adjoint=adjoint.__name__):
                    self._check(sim, direction, adjoint)

    def test_the_split_superposition_gradient_is_the_whole_one(self):
        for make in (_scalar, _elastic):
            with self.subTest(make=make.__name__):
                self._check(*make(4), superposition_sensitivity)

    def test_checkpoints_agree_on_one_segment(self):
        adjoint = functools.partial(reconstruction_sensitivity, checkpoints=2)
        for absorber in ("sponge", "pml"):
            with self.subTest(absorber=absorber):
                # the split strip is smaller, so the segment and its reversal differ
                self._check(*_elastic(4, absorber), adjoint, rtol=1e-4)

    def test_a_variant_without_a_split_refuses_devices(self):
        sim, _ = _scalar()
        source, sensors, gamma = _experiment(sim, None)
        with self.assertRaises(NotImplementedError):
            spectral_sensitivity(
                replace(sim, devices=DEVICES), source, gamma, sensors.nodes, None, 1.0
            )

    def test_an_error_in_one_subdomain_reaches_the_caller(self):
        sim, _ = _scalar()
        source, sensors, gamma = _experiment(sim, None)

        def objective(traces):
            raise ArithmeticError("the leader fails, the others wait at its barrier")

        with self.assertRaises(ArithmeticError):
            sensitivity(
                replace(sim, devices=DEVICES), source, gamma, sensors.nodes, objective
            )

    def test_a_halo_wider_than_a_share_is_refused(self):
        sim, _ = _scalar(8, Nx=(120, 64))
        source = point_source(sim, [[0.5, 0.3]], np.zeros(sim.N))
        gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
        with self.assertRaises(ValueError):
            simulate(replace(sim, devices=DEVICES), source, gamma)
