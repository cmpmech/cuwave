"""The two walls, and the per-face dispatch that picks between them.

A wall is the discrete method of images: the mirrored ghost adds an even image of
the source, the odd mirror an odd one. So the same run against the two conditions
gives `I + E` and `I - E` about the same free field `I`, and

    neumann + dirichlet == 2 * free

holds to round-off in the discretisation itself, not just in the PDE. That is
what `ReflectionTest` asserts, with `free` a third run of the identical setup
translated into a domain whose own walls are out of reach within the record, so
no arrival time has to be guessed anywhere.

`WallLayerTest` pins the argument `homogeneous_dirichlet_kernel` rests on: that writing
only the odd-mirrored ghost holds the wall node at zero without the kernel ever writing
it (docs/cuda_scalar.md derives it). It is not held there exactly: the fma contraction of
the two face fluxes leaves the rounding residual of a product, which then drives a stable
recursion on the wall layer, so the assertion is against machine epsilon rather than
against zero.

`PadTest` pins `pad_for_sponge`, the arithmetic a driver would otherwise repeat: the
four returns have to stay consistent with each other, since a `region` that disagrees
with the `origin` puts a transducer inside the layer without anything raising.

`SpongeTest` pins `sponge`, which is not a wall at all: it damps the nodes behind one,
so it can be held to an amplitude. The echo it leaves is a small fraction of the bare
wall's, against a reference whose own walls are out of reach within the record.
"""

import unittest
from dataclasses import replace

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # cupy missing or no GPU
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave.boundary import (
        Dirichlet,
        Neumann,
        canonical_boundary,
        define_boundary,
        pad_for_sponge,
        sponge,
    )
    from cuwave.scalar import ScalarWave
    from cuwave.signals import ricker
    from cuwave.wave import Source, compile_kernels, simulate, stable_dt

WAVESPEED = 1.0
DENSITY = 1.0
DX = 1.0 / 197  # fixed, so every domain below discretises the same medium
FREQUENCY = 8.0  # about 24 nodes per wavelength


def _sim(Nx, boundary, N, order=2, precision="float64"):
    dx = (DX,) * len(Nx)
    threads = (256,) if len(Nx) == 1 else (4, 32)
    return ScalarWave(
        Nx,
        dx,
        N,
        0.85 * stable_dt(dx, WAVESPEED, order),
        threads,
        precision=precision,
        space_order=order,
        wavespeed=WAVESPEED,
        density=DENSITY,
        boundary=boundary,
    )


def _pulse(sim, node, frequency=FREQUENCY):
    t = np.linspace(0.0, (sim.N - 1) * sim.dt, sim.N)
    signal = ricker(t, 1.0, frequency)
    return Source(
        cp.array([[i] for i in node], dtype=cp.int32),
        cp.asarray(signal[:, None], dtype=sim.dtype),
    )


def _run(sim, source_node, sensor_nodes, indicator=None, frequency=FREQUENCY):
    sensors = cp.array(
        [[node[d] for node in sensor_nodes] for d in range(sim.ndim)], dtype=cp.int32
    )
    if indicator is None:
        indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    _, um = simulate(
        sim, _pulse(sim, source_node, frequency), indicator, sensors=sensors
    )
    return um.get()


def _wall_slices(sim):
    # every wall layer of every axis: index 1 and Nx[d] - 2
    for d in range(sim.ndim):
        for index in (1, sim.Nx[d] - 2):
            sel = [slice(0, n) for n in sim.Nx]
            sel[d] = index
            yield tuple(sel)


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class ReflectionTest(unittest.TestCase):
    # 1D. The low wall is the only boundary either run can hear within NSTEPS
    NX, SOURCE, SENSOR, NSTEPS = 200, 40, 20, 250
    PAD = 300  # far enough that the free run hears no wall of its own

    def _trace(self, condition):
        sim = _sim((self.NX,), ((condition, Neumann),), self.NSTEPS)
        return _run(sim, (self.SOURCE,), [(self.SENSOR,)])[:, 0]

    def _free(self):
        sim = _sim((self.NX + 2 * self.PAD,), Neumann, self.NSTEPS)
        return _run(sim, (self.SOURCE + self.PAD,), [(self.SENSOR + self.PAD,)])[:, 0]

    def test_the_two_echoes_are_equal_and_opposite(self):
        neumann, dirichlet = self._trace(Neumann), self._trace(Dirichlet)
        free = self._free()
        scale = float(np.max(np.abs(free)))
        self.assertGreater(scale, 1e-6, "the sensor records essentially nothing")

        # an echo really did arrive, so the identity below is not trivially true
        echo = float(np.max(np.abs(neumann - free))) / scale
        print(f"  echo/incident = {echo:.2f}")
        self.assertGreater(echo, 0.1, "no reflection reached the sensor")

        # ... and it carries opposite signs off the two walls
        residual = float(np.max(np.abs(neumann + dirichlet - 2.0 * free))) / scale
        print(f"  |N + D - 2F| / |F| = {residual:.2e}")
        self.assertLess(residual, 1e-9)


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class WallLayerTest(unittest.TestCase):
    NSTEPS = 400

    def _field(self, Nx, boundary, order, precision):
        sim = _sim(Nx, boundary, self.NSTEPS, order=order, precision=precision)
        u = simulate(
            sim,
            _pulse(sim, (Nx[0] // 2,) * len(Nx)),
            cp.ones(sim.Nx_padded, dtype=sim.dtype),
        )
        scale = float(cp.max(cp.abs(u)))
        self.assertGreater(scale, 1e-6, "the pulse never developed")
        worst = max(float(cp.max(cp.abs(u[sel]))) for sel in _wall_slices(sim))
        return sim, worst / scale

    def test_wall_layer_stays_at_roundoff(self):
        for ndim in (1, 2):
            for order in (2, 4, 8):
                for precision in ("float32", "float64"):
                    with self.subTest(ndim=ndim, order=order, precision=precision):
                        sim, ratio = self._field(
                            (64,) * ndim, Dirichlet, order, precision
                        )
                        eps = float(np.finfo(sim.dtype).eps)
                        print(
                            f"  wall/field = {ratio:.2e} ({ratio / eps:8.1f} eps)  "
                            f"ndim={ndim} order={order} {precision}"
                        )
                        self.assertLess(ratio, 1e3 * eps)

    def test_the_reflecting_wall_layer_is_not_zero(self):
        # the control: without it the test above would pass on a field with no wall
        _, ratio = self._field((64,), Neumann, 2, "float64")
        self.assertGreater(ratio, 1e-3)


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class DispatchTest(unittest.TestCase):
    def test_canonical_boundary_expands_the_shorthands(self):
        self.assertEqual(canonical_boundary(None, 2), ((Neumann, Neumann),) * 2)
        self.assertEqual(
            canonical_boundary(Dirichlet, 3), ((Dirichlet, Dirichlet),) * 3
        )
        self.assertEqual(
            canonical_boundary((Dirichlet, (Neumann, Dirichlet)), 2),
            ((Dirichlet, Dirichlet), (Neumann, Dirichlet)),
        )
        with self.assertRaises(ValueError):
            canonical_boundary((Neumann,), 2)

    def test_one_launch_per_distinct_condition(self):
        # the default wall must still cost the single launch it costs today
        for boundary, expected in (
            (None, 1),
            (Dirichlet, 1),
            (((Dirichlet, Neumann), (Neumann, Neumann)), 2),
        ):
            with self.subTest(boundary=boundary):
                sim = _sim((32, 32), boundary, 1)
                kernels = compile_kernels(sim)
                asked = []

                class Counting:
                    def get_function(self, name):
                        asked.append(name)
                        return kernels.get_function(name)

                define_boundary(sim, Counting())
                self.assertEqual(len(asked), expected)

    def test_mixed_faces_2D(self):
        # x0 Dirichlet, x1 Neumann, a sensor at each, neither hearing the far wall
        Nx, source, low, high, N = (200, 8), 100, 20, 180, 250
        y = Nx[1] // 2
        sensors = [(low, y), (high, y)]

        def traces(boundary):
            return _run(_sim(Nx, boundary, N), (source, y), sensors)

        mixed = traces(((Dirichlet, Neumann), (Neumann, Neumann)))
        # same y walls, varying only in x, so each differs in the face its sensor misses
        neumann = traces(Neumann)
        dirichlet = traces(((Dirichlet, Dirichlet), (Neumann, Neumann)))
        scale = float(np.max(np.abs(neumann)))

        # the two references differ, so neither match below is vacuous
        self.assertGreater(np.max(np.abs(neumann - dirichlet)) / scale, 0.1)
        self.assertLess(np.max(np.abs(mixed[:, 0] - dirichlet[:, 0])) / scale, 1e-12)
        self.assertLess(np.max(np.abs(mixed[:, 1] - neumann[:, 1])) / scale, 1e-12)


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class PadTest(unittest.TestCase):
    NX, DX, THICKNESS = (100, 80), (0.01, 0.01), 0.1

    def _pad(self, faces=None, **kwargs):
        return pad_for_sponge(self.NX, self.DX, self.THICKNESS, faces, **kwargs)

    def test_the_thickness_becomes_a_node_count_off_the_finest_axis(self):
        _, width, _, _ = pad_for_sponge(self.NX, (0.01, 0.02), self.THICKNESS)
        self.assertEqual(width, 10)

    def test_every_face_grows_both_ends_of_every_axis(self):
        Nx, width, origin, region = self._pad()
        self.assertEqual(Nx, (120, 100))
        self.assertEqual(origin, (0.1, 0.1))
        self.assertEqual(region, (slice(11, 109), slice(11, 89)))
        self.assertEqual(width, 10)

    def test_only_the_named_faces_grow(self):
        # face 0 is the low side of axis 0, so axis 1 and the high side are untouched
        Nx, _, origin, region = self._pad(faces=(0,))
        self.assertEqual(Nx, (110, 80))
        self.assertEqual(origin, (0.1, 0.0))
        self.assertEqual(region, (slice(11, 109), slice(1, 79)))

    def test_the_region_selects_the_extent_that_was_asked_for(self):
        # the grown grid keeps the requested interior intact, ghost ring excluded
        for faces in (None, (0,), (1, 3), (0, 1, 2, 3)):
            with self.subTest(faces=faces):
                _, _, _, region = self._pad(faces=faces)
                sizes = tuple(r.stop - r.start for r in region)
                self.assertEqual(sizes, tuple(n - 2 for n in self.NX))

    def test_the_origin_places_the_region_where_its_slice_starts(self):
        for faces in (None, (0,), (1, 3)):
            with self.subTest(faces=faces):
                _, _, origin, region = self._pad(faces=faces)
                # node 1 is the origin of the grown grid, so the shift is one node in
                started = tuple((r.start - 1) * h for r, h in zip(region, self.DX))
                for got, want in zip(origin, started):
                    self.assertAlmostEqual(got, want, places=12)

    def test_the_bounds_are_checked(self):
        for kwargs in ({"faces": (4,)},):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self._pad(**kwargs)
        for thickness in (0.0, -1.0, 0.004):
            with self.subTest(thickness=thickness), self.assertRaises(ValueError):
                pad_for_sponge(self.NX, self.DX, thickness)


@unittest.skipUnless(HAS_CUDA, "requires CuPy and a CUDA device")
class SpongeTest(unittest.TestCase):
    # 2D. FINE shortens the wavelength, so several of them fit on an affordable grid
    NX, SOURCE, SENSOR, NSTEPS = 100, 50, 20, 600
    PAD = 170  # far enough that the reference hears no wall of its own
    FINE = 12.0
    WAVELENGTH = round(1.0 / FINE / DX)
    BETA = 0.1

    def _field(self, width, beta=None, **kwargs):
        sim = _sim((40, 40), None, 1)
        base = cp.ones(sim.Nx_padded, dtype=sim.dtype)
        return sim, sponge(
            sim, base, width, self.BETA if beta is None else beta, **kwargs
        )

    def test_the_field_is_zero_outside_the_layer(self):
        sim, field = self._field(5)
        # nodes 1..5 and 34..38 per axis are damped, so 6..33 stays lossless
        self.assertTrue(bool(cp.all(field[6:34, 6:34] == 0.0)))
        for edge in (field[0], field[39:], field[:, 0], field[:, 39:]):
            self.assertTrue(bool(cp.all(edge == 0.0)))

    def test_the_peak_matches_the_requested_beta(self):
        sim, field = self._field(5)
        # beta = d * dt / 2m, and m is the indicator itself under `derive_inertia`
        peak = float(field.max()) * sim.dt / 2.0
        print(f"  peak beta {peak:.4f} against {self.BETA}")
        self.assertAlmostEqual(peak, self.BETA, places=6)

    def test_only_the_named_faces_are_lined(self):
        _, field = self._field(5, faces=(0,))
        self.assertTrue(bool(cp.all(field[6:] == 0.0)))
        self.assertTrue(bool(cp.all(field[1:6, 1:39] > 0.0)))

    def test_the_bounds_are_checked(self):
        for kwargs in ({"beta": -0.1}, {"faces": (4,)}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self._field(5, **kwargs)
        for width in (0, 20):
            with self.subTest(width=width), self.assertRaises(ValueError):
                self._field(width)

    def _trace(self, pad, width):
        sim = _sim((self.NX + 2 * pad,) * 2, None, self.NSTEPS, precision="float32")
        indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
        if width:
            sim = replace(sim, damping=sponge(sim, indicator, width, self.BETA))
        return _run(
            sim,
            (self.SOURCE + pad,) * 2,
            [(self.SENSOR + pad, self.SOURCE + pad)],
            indicator=indicator,
            frequency=self.FINE,
        )[:, 0]

    def test_the_sponge_swallows_the_wall_echo(self):
        reference = self._trace(self.PAD, 0)
        scale = float(np.linalg.norm(reference))
        wall = float(np.linalg.norm(self._trace(0, 0) - reference)) / scale
        damped = float(
            np.linalg.norm(
                self._trace(2 * self.WAVELENGTH, 2 * self.WAVELENGTH) - reference
            )
            / scale
        )
        print(f"  residual: wall {wall:.3f}, sponge {damped:.4f}")
        # the wall echo dominates its own trace, so the comparison is not vacuous
        self.assertGreater(wall, 1.0, "no echo reached the sensor")
        self.assertLess(damped, wall / 20.0)


if __name__ == "__main__":
    unittest.main()
