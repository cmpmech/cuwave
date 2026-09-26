"""`march`, the graph replay every time loop runs through, and `Simulation.domain`.

Replaying a captured chunk has to be the plain loop exactly: same kernels, same order,
so the fields agree bit for bit. What can break is the staging around it, so each
test takes a step count that leaves a partial tail chunk and compares against
`CHUNK` patched past `N`, which forces the plain loop. The windows covered are the
three kinds: a plain input and output (`simulate`), an output with a loaded halo
(the `sensitivity` history), and reversed contiguous inputs (the strip replay of
`reconstruction_sensitivity`, with the damping that gives it a strip).

`DomainTest` pins what makes skipping tiles safe: the graded cells never read a node
outside the domain, so which tiles run, and what material lies outside, change nothing
bit for bit, and a domain covering everything is the plain scheme to round-off. At order 2 the operator
stays symmetric, which `source_sensitivity` against a finite difference checks, since
its adjoint recursion is the forward step itself; the wider orders are as symmetric
there as at the box walls, or more.

A `dirichlet` node opens its cell onto a zero at the cell midpoint, so a held row is
the method of images about that midpoint: at order 2 it matches the unmasked run with
a mirrored, negated source to round-off.
"""

import unittest
from unittest import mock

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # cupy missing or no GPU
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave import wave
    from cuwave.boundary import sponge
    from cuwave.elastic import ElasticWave
    from cuwave.scalar import ScalarWave
    from cuwave.sensitivity import (
        l2_misfit,
        reconstruction_sensitivity,
        sensitivity,
        source_sensitivity,
    )
    from cuwave.wave import CHUNK, Source, grid_coords, simulate, stable_dt

NX = (40, 48)
ORIGINAL_TILES = wave.domain_tiles if HAS_CUDA else None


def _setup(damped=False, precision="float32", order=4):
    dx = tuple(1.0 / (n - 3) for n in NX)
    N = 2 * CHUNK + 7
    sim = ScalarWave(
        NX,
        dx,
        N,
        0.5 * stable_dt(dx, 1.0, 4),
        (4, 32),
        space_order=order,
        precision=precision,
        wavespeed=1.0,
        density=1.0,
    )
    indicator = 1.0 + 0.2 * cp.random.RandomState(0).rand(*sim.Nx_padded)
    indicator = indicator.astype(sim.dtype)
    if damped:
        sim.damping = sponge(sim, indicator, 6, 0.05)
    signal = np.sin(0.3 * np.arange(N))[:, None]
    source = Source(cp.array([[20], [24]], cp.int32), cp.asarray(signal, sim.dtype))
    sensors = cp.array([[10, 10, 12], [8, 9, 30]], cp.int32)
    return sim, source, indicator, sensors


def _disk_setup(precision="float32", order=4):
    """`_setup` inside a disk, run long enough to reach its wall, sensors inside."""
    sim, source, indicator, _ = _setup(precision=precision, order=order)
    sim.N = 6 * CHUNK + 7
    source.signal = cp.asarray(np.sin(0.3 * np.arange(sim.N))[:, None], sim.dtype)
    X = grid_coords(sim.Nx, sim.dx)
    sim.domain = (X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 <= 0.42**2
    sensors = cp.array([[12, 20, 28, 7], [14, 36, 20, 24]], cp.int32)
    return sim, source, indicator, sensors


def _every_tile(sim, cells):
    return ORIGINAL_TILES(sim, cp.full_like(cells, 1)) | (1 << 30)


def _plain(run):
    with mock.patch("cuwave.wave.CHUNK", 10**6):
        return run()


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class MarchTest(unittest.TestCase):
    def test_simulate_replays_the_plain_loop(self):
        sim, source, indicator, sensors = _setup()
        replayed = simulate(sim, source, indicator, sensors)
        plain = _plain(lambda: simulate(sim, source, indicator, sensors))
        for a, b in zip(replayed, plain):
            np.testing.assert_array_equal(a.get(), b.get())

    def test_a_staged_history_replays_the_plain_loop(self):
        sim, source, indicator, sensors = _setup()
        objective = l2_misfit(cp.zeros((sim.N, 3), sim.dtype))
        replayed = sensitivity(sim, source, indicator, sensors, objective)
        plain = _plain(lambda: sensitivity(sim, source, indicator, sensors, objective))
        for name in replayed[1]:
            np.testing.assert_array_equal(replayed[1][name].get(), plain[1][name].get())

    def test_a_replayed_strip_matches_the_plain_loop(self):
        sim, source, indicator, sensors = _setup(damped=True)
        objective = l2_misfit(cp.zeros((sim.N, 3), sim.dtype))

        def run():
            return reconstruction_sensitivity(
                sim, source, indicator, sensors, objective
            )

        replayed, plain = run(), _plain(run)
        self.assertGreater(replayed[3]["strip"], 0)
        for name in replayed[1]:
            np.testing.assert_array_equal(replayed[1][name].get(), plain[1][name].get())


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class DomainTest(unittest.TestCase):
    def test_skipping_tiles_changes_nothing(self):
        sim, source, indicator, sensors = _disk_setup()
        skipped = simulate(sim, source, indicator, sensors)
        with mock.patch("cuwave.wave.domain_tiles", _every_tile):
            every = simulate(sim, source, indicator, sensors)
        for a, b in zip(skipped, every):
            np.testing.assert_array_equal(a.get(), b.get())

    def test_skipping_tiles_changes_nothing_around_a_held_obstacle(self):
        sim, source, indicator, sensors = _disk_setup()
        X = grid_coords(sim.Nx, sim.dx)
        sim.dirichlet = (X[0] - 0.62) ** 2 + (X[1] - 0.35) ** 2 <= 0.1**2
        skipped = simulate(sim, source, indicator, sensors)
        with mock.patch("cuwave.wave.domain_tiles", _every_tile):
            every = simulate(sim, source, indicator, sensors)
        for a, b in zip(skipped, every):
            np.testing.assert_array_equal(a.get(), b.get())

    def test_a_held_row_is_the_method_of_images(self):
        # the box mirrors about row 19.5, the midpoint the held row 20 puts its zero at
        dx = (0.02, 0.02)
        sim = ScalarWave(
            (40, 36),
            dx,
            150,
            0.9 * stable_dt(dx, 1.0),
            (4, 32),
            precision="float64",
            wavespeed=1.0,
            density=1.0,
        )
        signal = np.sin(0.3 * np.arange(sim.N))
        one = Source(cp.array([[12], [15]], cp.int32), cp.asarray(signal[:, None]))
        pair = Source(
            cp.array([[12, 27], [15, 15]], cp.int32),
            cp.asarray(np.stack([signal, -signal], 1)),
        )
        indicator = cp.ones(sim.Nx_padded)
        imaged = simulate(sim, pair, indicator)
        sim.dirichlet = cp.zeros(sim.Nx_padded, dtype=cp.bool_)
        sim.dirichlet[20] = True
        held = simulate(sim, one, indicator)
        below = (slice(1, 20), slice(1, 35))
        scale = float(cp.abs(imaged[below]).max())
        np.testing.assert_allclose(
            held[below].get(), imaged[below].get(), atol=1e-12 * scale
        )

    def test_the_material_outside_is_never_read(self):
        sim, source, indicator, sensors = _disk_setup()
        voided = cp.where(sim.domain, indicator, sim.dtype(1e-4))
        filled = cp.where(sim.domain, indicator, sim.dtype(1.0))
        a = simulate(sim, source, voided, sensors)
        b = simulate(sim, source, filled, sensors)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.get(), y.get())

    def test_a_domain_covering_everything_is_the_plain_scheme(self):
        sim, source, indicator, sensors = _setup()
        plain = simulate(sim, source, indicator, sensors)[1]
        sim.domain = cp.ones(sim.Nx_padded, dtype=cp.bool_)
        covered = simulate(sim, source, indicator, sensors)[1]
        # to round-off only: the domain build contracts its fmas differently
        scale = float(cp.abs(plain).max())
        np.testing.assert_allclose(covered.get(), plain.get(), atol=1e-5 * scale)

    def test_the_graded_operator_is_its_own_adjoint(self):
        sim, source, indicator, sensors = _disk_setup("float64", order=2)
        observed = cp.zeros((sim.N, sensors.shape[1]), sim.dtype)
        objective = l2_misfit(observed)
        gradient = source_sensitivity(sim, source, indicator, sensors, objective)[1]
        direction = cp.random.RandomState(1).standard_normal(source.signal.shape)
        h = 1e-3

        def cost(signal):
            shot = Source(source.position, signal)
            return objective(simulate(sim, shot, indicator, sensors)[1])[0]

        plus, minus = source.signal + h * direction, source.signal - h * direction
        reference = (cost(plus) - cost(minus)) / (2 * h)
        self.assertAlmostEqual(float(cp.sum(gradient * direction)) / reference, 1, 8)

    def test_a_sensor_outside_the_domain_raises(self):
        sim, source, indicator, _ = _disk_setup()
        with self.assertRaises(ValueError):
            simulate(sim, source, indicator, cp.array([[2], [2]], cp.int32))

    def test_a_source_on_a_held_node_raises(self):
        sim, source, indicator, sensors = _setup()
        sim.dirichlet = cp.zeros(sim.Nx_padded, dtype=cp.bool_)
        sim.dirichlet[tuple(source.position[:, 0].tolist())] = True
        with self.assertRaises(ValueError):
            simulate(sim, source, indicator, sensors)

    def test_a_source_outside_the_domain_raises(self):
        sim, source, indicator, sensors = _setup()
        sim.domain = cp.zeros(sim.Nx_padded, dtype=cp.bool_)
        with self.assertRaises(ValueError):
            simulate(sim, source, indicator, sensors)

    def test_the_material_gradients_refuse_a_domain(self):
        sim, source, indicator, sensors = _disk_setup()
        objective = l2_misfit(cp.zeros((sim.N, 3), sim.dtype))
        with self.assertRaises(NotImplementedError):
            sensitivity(sim, source, indicator, sensors, objective)

    def test_an_equation_without_the_graded_kernel_refuses_a_domain(self):
        sim = ElasticWave(
            NX,
            (0.1, 0.1),
            10,
            1e-3,
            (4, 32),
            density=1.0,
            wavespeed_p=2.0,
            wavespeed_s=1.0,
        )
        sim.domain = cp.ones(sim.Nx_padded, dtype=cp.bool_)
        with self.assertRaises(ValueError):
            sim.compile_flags


if __name__ == "__main__":
    unittest.main()
