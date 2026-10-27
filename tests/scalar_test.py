"""The two step kernels of the pressure family, which must be one operator.

A wide unmasked 2D step on a large grid streams along axis 0 (`USE_STREAM`): the axis-0
taps queue in registers while the row sits in shared memory. Everywhere else the step is
the per-thread kernel, so the tests force each in turn rather than trust the switch. Both take the taps in the same order, so they agree bit for
bit at order 2 and to the last bit of a contracted fma above it. What can break is the
queue across chunk boundaries, the tile halo across block boundaries and past a narrow
tile, the ghost rows the queue reads, and the damping and inertia paths, each pinned here
against the per-thread kernel.

A 2D float32 run on a small or a DRAM-bound grid advances several steps a launch
(`define_blocked_step`), the excitation, the ghost refill and the record inside the tile.
It claims the plain run bit for bit, so it is compared with `array_equal` over orders,
boundaries, damping, inertia, tiles, step counts and sources and sensors on the wall.
A 2D run past L2 marches runs of columns along axis 0 with the same claim, checked on
runs and chunks small enough that a test grid holds many, the last of each one node.
A 3D order-2 run past L2 marches columns of tiles along axis 0 with the same claim,
checked on a grid whose last tile and last chunk of planes hold one node each.

Where no mask, pml or streaming stands in the way, up to order 4, the per-thread step
writes the face ghosts from their mirrors two nodes in, and in 3D adds the sources its
block holds, so `simulate` drops the refill launch, and in 3D the excitation's too. It
claims the launches apart bit for bit, so it is compared with them, in 1D to 3D, over
faces and dampings, with a 3D source on a mirror and sensors on the ghosts.
"""

import copy
import itertools
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # cupy missing or no GPU
    HAS_CUDA = False

if HAS_CUDA:
    from cuwave import scalar
    from cuwave.boundary import Dirichlet, Neumann, Pml, sponge
    from cuwave.scalar import AcousticWave, ScalarWave
    from cuwave.signals import ricker
    from cuwave.utils import point_source
    from cuwave.wave import Source, compile_kernels, simulate, stable_dt

NX = (150, 77)  # three chunks of axis 0, and a width no tile divides


# -------------------------------------- helpers --------------------------------------
def _scalar(order, precision="float64", boundary=None, threads=(4, 32), N=1, Nx=NX):
    dx = (0.1, 0.1)
    return ScalarWave(
        Nx,
        dx,
        N,
        0.5 * stable_dt(dx, 1.0, order),
        threads,
        precision=precision,
        space_order=order,
        wavespeed=1.0,
        density=1.0,
        boundary=boundary,
    )


def _step(sim, indicator, fields):
    """One step of `fields` = (u0, u1, u2) under the kernel `sim` compiles."""
    mat = sim.build_materials(indicator.copy())
    u0, u1, u2 = (f.copy() for f in fields)
    return sim.define_step(compile_kernels(sim), mat)(u0, u1, u2)


def _both(sim, indicator, fields):
    """The interior after one step, streamed and per thread, whatever `streams` picks."""
    steps = []
    for streams in (True, False):
        with mock.patch.object(type(sim), "streams", streams):
            steps.append(_step(sim, indicator, fields))
    interior = tuple(slice(1, n - 1) for n in sim.Nx)
    return steps[0][interior], steps[1][interior]


def _random(sim, seed):
    rng = np.random.default_rng(seed)
    indicator = cp.asarray(0.3 + rng.random(sim.Nx_padded), dtype=sim.dtype)
    fields = cp.asarray(rng.standard_normal((3, *sim.Nx_padded)), dtype=sim.dtype)
    return indicator, fields


# ---------------------------------- the two kernels ----------------------------------
@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class StreamingTest(unittest.TestCase):
    def _assert_same(self, sim, a, b):
        scale = float(cp.abs(b).max())
        ulps = 0.0 if sim.space_order == 2 else 8 * float(np.finfo(sim.dtype).eps)
        self.assertLessEqual(float(cp.abs(a - b).max()), ulps * scale)

    def test_the_streaming_step_is_the_per_thread_step(self):
        for order in (2, 4, 8, 16):
            for precision in ("float32", "float64"):
                for boundary in (None, Dirichlet):
                    with self.subTest(order=order, precision=precision, b=boundary):
                        sim = _scalar(order, precision, boundary)
                        self._assert_same(sim, *_both(sim, *_random(sim, order)))

    def test_a_tile_narrower_than_its_halo(self):
        # order 16 reaches 8 nodes, past a tile of 4 threads
        sim = _scalar(16, threads=(1, 4))
        self._assert_same(sim, *_both(sim, *_random(sim, 3)))

    def test_the_damped_update_and_a_formed_inertia(self):
        dx = (0.1, 0.1)
        sim = AcousticWave(
            NX,
            dx,
            1,
            1e-5,
            (4, 32),
            precision="float64",
            space_order=4,
            rho1=1.2,
            rho2=2600.0,
            kappa1=1.4e5,
            kappa2=6.9e8,
        )
        indicator, fields = _random(sim, 4)
        sim = replace(sim, damping=sponge(sim, indicator, 10, 0.1))
        self._assert_same(sim, *_both(sim, indicator, fields))

    def test_only_a_wide_unmasked_2d_step_on_a_full_grid_streams(self):
        dx = (0.1, 0.1)
        sim = ScalarWave(
            (4098, 2050),
            dx,
            1,
            1e-3,
            (4, 128),
            space_order=8,
            wavespeed=1.0,
            density=1.0,
        )
        self.assertTrue(sim.streams)
        self.assertFalse(replace(sim, space_order=4).streams)
        self.assertFalse(replace(sim, Nx=(130, 130)).streams)
        sim.domain = cp.ones(sim.Nx_padded, dtype=cp.bool_)
        self.assertFalse(sim.streams)
        sim = ScalarWave(
            (200, 200, 200),
            (0.1,) * 3,
            1,
            1e-3,
            (1, 4, 32),
            space_order=8,
            wavespeed=1.0,
            density=1.0,
        )
        self.assertFalse(sim.streams)

    def test_a_run_streams_like_the_per_thread_loop(self):
        # every step, the ghost kernels and the excitation in between
        sim = _scalar(8, N=200)
        t = np.arange(sim.N) * sim.dt
        source = point_source(sim, (3.0, 2.0), ricker(t, 1.0, 2.0))
        indicator, _ = _random(sim, 5)
        runs = []
        for streams in (True, False):
            with mock.patch.object(type(sim), "streams", streams):
                runs.append(simulate(sim, source, indicator))
        streamed, per_thread = runs
        scale = float(cp.abs(per_thread).max())
        self.assertGreater(scale, 0.0)
        self.assertLess(float(cp.abs(streamed - per_thread).max()), 1e-12 * scale)


# ------------------------------------ blocked steps -----------------------------------
def _shot(sim, seed):
    """Three sources, two beside a wall, and sensors in the interior, on a wall and in ghosts."""
    t = np.arange(sim.N) * sim.dt
    signal = np.stack(
        [ricker(t, 1.0, 2.0), -ricker(t, 0.7, 2.5), ricker(t, 1.3, 1.0)], 1
    )
    n = sim.Nx
    position = [[n[0] // 3, 2 * n[0] // 3, n[0] - 3], [n[1] // 2, 1, n[1] - 3]]
    position = cp.array(position, dtype=cp.int32)
    sensors = cp.array(
        [[1, 2, n[0] - 2, 0, 5, 70, n[0] - 1], [1, n[1] - 2, 3, 4, n[1] - 1, 40, 9]],
        dtype=cp.int32,
    )
    source = Source(position, cp.asarray(signal, dtype=sim.dtype))
    return source, _random(sim, seed)[0], sensors


def _runs(sim, seed, sensed=True, **blocks):
    """The (field, record) of a run blocked as `blocks` patch, then of the plain run."""
    source, indicator, sensors = _shot(sim, seed)
    sensors = sensors if sensed else None
    # the small tier of every order on any GPU
    small = {"SMALL_BLOCKS": scalar.SMALL_BLOCKS, "SMALL_SMS": (0,) * 4}
    small["SMALL_WAVES"] = (*scalar.SMALL_WAVES[:2], 100, 100)
    with mock.patch.multiple(scalar, **{**small, **blocks}):
        mat = sim.build_materials(indicator)
        assert sim.define_blocked_step(mat, source, sensors) is not None
        blocked = simulate(sim, source, indicator, sensors=sensors)
    with mock.patch.object(type(sim), "define_blocked_step", lambda *a, **k: None):
        plain = simulate(sim, source, indicator, sensors=sensors)
    return (blocked, plain) if sensed else ((blocked,), (plain,))


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class BlockedTest(unittest.TestCase):
    def _assert_bitwise(self, blocked, plain):
        self.assertGreater(float(cp.abs(plain[0]).max()), 0.0)
        for a, b in zip(blocked, plain):
            self.assertTrue(bool(cp.array_equal(a, b)))

    def test_a_run_without_sensors(self):
        sim = _scalar(2, "float32", Dirichlet, N=30)
        self._assert_bitwise(*_runs(sim, 9, sensed=False))

    def test_a_blocked_run_is_the_plain_run_bit_for_bit(self):
        mixed = ((Neumann, Dirichlet), (Dirichlet, Neumann))
        for order in (2, 4, 6, 8):
            for boundary in (None, Dirichlet, mixed):
                for N in (1, 5, 61):
                    with self.subTest(order=order, boundary=boundary, N=N):
                        sim = _scalar(order, "float32", boundary, N=N)
                        self._assert_bitwise(*_runs(sim, order))

    def test_every_tile_and_step_count(self):
        tiles = (((32, 64), 8), ((32, 64), 4), ((64, 64), 16), ((32, 96), 8))
        orders = (2, 4, 6, 8)
        for order, tile, steps in itertools.product(orders, tiles, (1, 2, 3, 4, 6, 12)):
            if min(tile[0]) - 2 * steps * order // 2 < 8 or steps * order // 2 < 2:
                continue
            with self.subTest(order=order, tile=tile, steps=steps):
                sim = _scalar(order, "float32", Dirichlet, N=50)
                blocks = (((tile[0], tile[1], steps),),) * 4
                self._assert_bitwise(*_runs(sim, 6, SMALL_BLOCKS=blocks))

    def test_a_copied_simulation(self):
        # a deep copy holds equal, not identical, conditions
        sim = copy.deepcopy(_scalar(2, "float32", Dirichlet, N=20))
        self._assert_bitwise(*_runs(sim, 3))

    def test_a_last_core_of_one_node(self):
        # cores (24, 56): each axis's last core holds the node beside the ghost alone
        sim = _scalar(2, "float32", Dirichlet, N=24, Nx=(147, 115))
        blocks = ((((32, 64), 8, 4),),) * 2
        self._assert_bitwise(*_runs(sim, 5, SMALL_BLOCKS=blocks))

    def test_the_damped_update_and_a_formed_inertia(self):
        sim = AcousticWave(
            NX,
            (0.1, 0.1),
            40,
            1e-5,
            (4, 32),
            space_order=2,
            rho1=1.2,
            rho2=2600.0,
            kappa1=1.4e5,
            kappa2=6.9e8,
        )
        damped = replace(sim, damping=sponge(sim, _random(sim, 7)[0], 10, 0.1))
        for each in (sim, damped):
            with self.subTest(damped=each.damping is not None):
                self._assert_bitwise(*_runs(each, 7))

    def test_what_the_plain_step_keeps(self):
        sim = _scalar(2, "float32", N=6)
        source, indicator, sensors = _shot(sim, 8)
        mat = sim.build_materials(indicator)
        self.assertIsNotNone(sim.define_blocked_step(mat, source, sensors))
        refused = {
            "float64": _scalar(2, "float64", N=6),
            "order 10": _scalar(10, "float32", N=6),
            "pml": replace(sim, pml=Pml(5, 1.0)),
            "sensor twice": sim,
        }
        twice = cp.concatenate([sensors, sensors[:, :1]], axis=1)
        for name, each in refused.items():
            with self.subTest(name=name):
                mat = each.build_materials(indicator.astype(each.dtype))
                taken = twice if name == "sensor twice" else sensors
                self.assertIsNone(each.define_blocked_step(mat, source, taken))
        # fields L2 serves, past the waves launches bound: the plain step's ground
        middle = ScalarWave(
            (1024, 1024),
            (0.1, 0.1),
            6,
            1e-3,
            (4, 32),
            space_order=4,
            wavespeed=1.0,
            density=1.0,
        )
        self.assertIsNone(middle.define_blocked_step(mat, source, None))


def _shot_walls(sim, seed):
    """A source beside each wall and a corner, sensors along the walls and inside."""
    n = sim.Nx
    signal = np.random.default_rng(seed).standard_normal((sim.N, 6))
    position = [[2, n[0] // 2, n[0] - 3, n[0] // 3, 1, n[0] // 2]]
    position += [[n[1] // 3, 2, n[1] // 2, n[1] - 3, 1, n[1] // 2]]
    near = [[a, 3] for a in range(1, 9)] + [[1, n[1] - 2], [n[0] - 2, 1]]
    sensors = np.array(near + [[n[0] - 2, n[1] - 2], [n[0] // 2, 40]]).T
    source = Source(cp.array(position, dtype=cp.int32), cp.asarray(signal, sim.dtype))
    return source, _random(sim, seed)[0], cp.array(sensors, dtype=cp.int32)


def _marched_runs(sim, seed, rows):
    """The (field, record) of a 2D run marched in `rows`, then of the plain run."""
    source, indicator, sensors = _shot_walls(sim, seed)
    with mock.patch.multiple(scalar, MARCHED_ROWS=rows, LARGE_L2=0):
        mat = sim.build_materials(indicator)
        assert sim._define_marched_rows(mat, source, sensors) is not None
        marched = simulate(sim, source, indicator, sensors=sensors)
    with mock.patch.object(type(sim), "define_blocked_step", lambda *a, **k: None):
        plain = simulate(sim, source, indicator, sensors=sensors)
    return marched, plain


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class MarchedRowsTest(unittest.TestCase):
    ROWS = ((64, 4, 16), (64, 3, 16))  # runs and chunks a test grid holds many of

    def _assert_bitwise(self, marched, plain):
        self.assertGreater(float(cp.abs(plain[0]).max()), 0.0)
        for a, b in zip(marched, plain):
            self.assertTrue(bool(cp.array_equal(a, b)))

    def test_a_marched_run_is_the_plain_run_bit_for_bit(self):
        mixed = ((Neumann, Dirichlet), (Dirichlet, Neumann))
        for order in (2, 4):
            for boundary in (None, Dirichlet, mixed):
                for N in (5, 61):
                    with self.subTest(order=order, boundary=boundary, N=N):
                        sim = _scalar(order, "float32", boundary, N=N, Nx=(301, 257))
                        self._assert_bitwise(*_marched_runs(sim, order, self.ROWS))

    def test_a_last_run_and_chunk_of_one_node(self):
        # cores of 56 columns and chunks of 16 rows: the last of each holds one node
        sim = _scalar(2, "float32", Dirichlet, N=24, Nx=(147, 115))
        self._assert_bitwise(*_marched_runs(sim, 5, self.ROWS))

    def test_the_damped_update_and_a_formed_inertia(self):
        for order in (2, 4):
            sim = AcousticWave(
                (301, 257),
                (0.1, 0.1),
                40,
                1e-5,
                (4, 32),
                space_order=order,
                rho1=1.2,
                rho2=2600.0,
                kappa1=1.4e5,
                kappa2=6.9e8,
            )
            damped = replace(sim, damping=sponge(sim, _random(sim, 7)[0], 30, 0.1))
            for each in (sim, damped):
                with self.subTest(order=order, damped=each.damping is not None):
                    self._assert_bitwise(*_marched_runs(each, 7, self.ROWS))

    def test_five_columns_take_the_tiles(self):
        # the middle column mirrors both ghost columns, which one mirror cannot write
        sim = _scalar(
            2,
            "float32",
            ((Neumann, Dirichlet), (Neumann, Dirichlet)),
            N=30,
            Nx=(301, 5),
        )
        signal = np.random.default_rng(4).standard_normal((sim.N, 3))
        position = cp.array([[2, 150, 298], [1, 3, 2]], dtype=cp.int32)
        source = Source(position, cp.asarray(signal, dtype=sim.dtype))
        sensors = cp.array([[5, 100, 200, 299], [1, 2, 3, 3]], dtype=cp.int32)
        indicator = _random(sim, 4)[0]
        mat = sim.build_materials(indicator)
        with mock.patch.multiple(scalar, MARCHED_ROWS=self.ROWS, LARGE_L2=0):
            self.assertIsNone(sim._define_marched_rows(mat, source, sensors))
            blocked = simulate(sim, source, indicator, sensors=sensors)
        with mock.patch.object(type(sim), "define_blocked_step", lambda *a, **k: None):
            plain = simulate(sim, source, indicator, sensors=sensors)
        self._assert_bitwise(blocked, plain)

    def test_a_sensor_on_a_ghost_takes_the_tiles(self):
        sim = _scalar(2, "float32", N=6, Nx=(301, 257))
        source, indicator, sensors = _shot(sim, 8)
        mat = sim.build_materials(indicator)
        with mock.patch.object(scalar, "LARGE_L2", 0):
            self.assertIsNone(sim._define_marched_rows(mat, source, sensors))
            self.assertIsNotNone(sim.define_blocked_step(mat, source, sensors))


def _scalar3d(boundary, N, Nx=(259, 243, 243)):
    # cores (24, 24) and 64 planes: the last of each holds the node beside the ghost
    dx = (0.1, 0.1, 0.1)
    dt = 0.5 * stable_dt(dx, 1.0, 2)
    common = dict(space_order=2, boundary=boundary)
    return ScalarWave(Nx, dx, N, dt, (1, 16, 32), wavespeed=1.0, density=1.0, **common)


def _shot3d(sim, seed):
    """Sources beside two corners, a wall and inside, sensors by the walls and along a column."""
    n = sim.Nx
    signal = np.random.default_rng(seed).standard_normal((sim.N, 4))
    position = [[2, n[0] - 3, n[0] // 2, 100], [2, n[1] - 3, n[1] // 3, 120]]
    position += [[3, n[2] - 3, 7, 120]]
    column = [[a, 5, n[2] // 2] for a in range(1, 12)]
    near = [[1, 1, 1], [n[0] - 2, n[1] - 2, n[2] - 2], [n[0] - 2, 2, 50], [70, 1, 9]]
    sensors = np.array(column + near + [[66, 30, n[2] - 2], [3, 2, 3]]).T
    source = Source(cp.array(position, dtype=cp.int32), cp.asarray(signal, sim.dtype))
    # inside the two phases, which also keeps the scalar density positive
    indicator = 0.25 + 0.5 * (_random(sim, seed)[0] - 0.3)
    return source, indicator, cp.array(sensors, dtype=cp.int32)


def _runs3d(sim, seed):
    """The (field, record) of the marched run, then of the plain run."""
    source, indicator, sensors = _shot3d(sim, seed)
    mat = sim.build_materials(indicator)
    assert sim.define_blocked_step(mat, source, sensors) is not None
    blocked = simulate(sim, source, indicator, sensors=sensors)
    with mock.patch.object(type(sim), "define_blocked_step", lambda *a, **k: None):
        plain = simulate(sim, source, indicator, sensors=sensors)
    return blocked, plain


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class MarchedTest(unittest.TestCase):
    def _assert_bitwise(self, blocked, plain):
        self.assertGreater(float(cp.abs(plain[0]).max()), 0.0)
        for a, b in zip(blocked, plain):
            self.assertTrue(bool(cp.array_equal(a, b)))

    def test_a_marched_run_is_the_plain_run_bit_for_bit(self):
        mixed = ((Neumann, Dirichlet), (Dirichlet, Neumann), (Dirichlet, Dirichlet))
        for boundary in (None, mixed):
            for N in (3, 41):
                with self.subTest(boundary=boundary, N=N):
                    self._assert_bitwise(*_runs3d(_scalar3d(boundary, N), 3))

    def test_no_memory_for_the_second_pair_steps_plainly(self):
        sim = _scalar3d(None, 9)
        source, indicator, sensors = _shot3d(sim, 2)
        plain = _runs3d(sim, 2)[1]
        zeros_like = cp.zeros_like

        def full(a, *args, **kwargs):
            if a.shape == (2, *sim.Nx_padded):
                raise cp.cuda.memory.OutOfMemoryError(a.nbytes, 0)
            return zeros_like(a, *args, **kwargs)

        with mock.patch.object(cp, "zeros_like", full):
            starved = simulate(sim, source, indicator, sensors=sensors)
        self._assert_bitwise(starved, plain)

    def test_the_damped_update(self):
        sim = _scalar3d(Dirichlet, 10)
        damped = replace(sim, damping=sponge(sim, _random(sim, 7)[0], 12, 0.1))
        self._assert_bitwise(*_runs3d(damped, 5))

    def test_a_formed_inertia(self):
        sim = AcousticWave(
            (259, 243, 243),
            (0.1, 0.1, 0.1),
            10,
            1e-5,
            (1, 16, 32),
            space_order=2,
            boundary=Dirichlet,
            rho1=1.2,
            rho2=2600.0,
            kappa1=1.4e5,
            kappa2=6.9e8,
        )
        self._assert_bitwise(*_runs3d(sim, 4))

    def test_what_the_plain_step_keeps(self):
        sim = _scalar3d(None, 4)
        source, indicator, sensors = _shot3d(sim, 8)
        refused = {
            "order 4": replace(sim, space_order=4),
            "in L2": _scalar3d(None, 4, (40, 40, 40)),
            "a sensor on a ghost": sim,
        }
        for name, each in refused.items():
            with self.subTest(name=name):
                mat = each.build_materials(indicator)
                taken = sensors.copy()
                if name == "a sensor on a ghost":
                    taken[0, 0] = 0
                if name == "in L2":
                    taken = None
                self.assertIsNone(each.define_blocked_step(mat, source, taken))


# ---------------------------------- the folded step ----------------------------------
def _folded_runs(sim, seed):
    """The (field, record) of a run whose step takes the ghosts (and in 3D the sources), then of one launching them apart."""
    n = np.asarray(sim.Nx)
    points = [n // 3, n // 2]
    if sim.ndim == 3:
        points.append(np.r_[2, n[1:] // 2])  # on a mirror, its ghost takes the source
    position = cp.asarray(np.stack(points).T, dtype=cp.int32)
    signal = np.random.default_rng(seed).standard_normal((sim.N, len(points)))
    source = Source(position, cp.asarray(signal, dtype=sim.dtype))
    # the ghosts, mirrors and nodes between along the diagonal, a face ghost, a mirror
    rows = [np.full(sim.ndim, k) for k in (0, 1, 2, 3)]
    rows += [n - k for k in (1, 2, 3, 4)] + [
        np.r_[0, n[1:] // 2],
        np.r_[n[:1] - 3, 2 * n[1:] // 3],
    ]
    sensors = cp.asarray(np.stack(rows).T, dtype=cp.int32)
    indicator = _random(sim, seed)[0]
    runs = []
    blocked = mock.patch.object(type(sim), "define_blocked_step", lambda *a, **k: None)
    with blocked, mock.patch.object(scalar, "FOLD_SMS", 0):
        mat = sim.build_materials(indicator)
        assert sim.define_folded_step(compile_kernels(sim), mat, source) is not None
        runs.append(simulate(sim, source, indicator, sensors=sensors))
        with mock.patch.object(scalar.PressureWave, "_refills_in_step", False):
            runs.append(simulate(sim, source, indicator, sensors=sensors))
    return runs


@unittest.skipUnless(HAS_CUDA, "needs a CUDA device")
class FoldedStepTest(unittest.TestCase):
    def test_a_folded_run_is_the_launched_run_bit_for_bit(self):
        mixed = (Neumann, Dirichlet), (Dirichlet, Neumann), (Dirichlet, Dirichlet)
        grids = {1: (61,), 2: (37, 45), 3: (21, 18, 35)}
        threads = {1: (32,), 2: (4, 32), 3: (1, 16, 32)}
        for ndim, Nx in grids.items():
            for boundary, precision in itertools.product(
                (None, mixed[:ndim]), ("float32", "float64")
            ):
                dx = (0.1,) * ndim
                sim = ScalarWave(
                    Nx,
                    dx,
                    53,  # past two graphed chunks, so the windows are staged
                    0.5 * stable_dt(dx, 1.0, 2),
                    threads[ndim],
                    precision=precision,
                    wavespeed=1.0,
                    density=1.0,
                    boundary=boundary,
                )
                for damped in (False, True):
                    if damped:
                        indicator = _random(sim, 5)[0]
                        sim = replace(sim, damping=sponge(sim, indicator, 4, 0.5))
                    with self.subTest(
                        ndim=ndim, boundary=boundary, precision=precision, damped=damped
                    ):
                        folded, launched = _folded_runs(sim, 7)
                        self.assertGreater(float(cp.abs(launched[0]).max()), 0.0)
                        for a, b in zip(folded, launched):
                            self.assertTrue(bool(cp.array_equal(a, b)))

    def test_what_keeps_the_launches_apart(self):
        sim = _scalar(2, Nx=(37, 45))
        cube = ScalarWave(
            (21, 18, 35), (0.1,) * 3, 1, 1e-3, (1, 16, 32), wavespeed=1.0, density=1.0
        )
        kept = {
            "a source on a mirror": (sim, [[2], [20]]),
            "a source on a face ghost": (sim, [[0], [20]]),
            "a source on an edge ghost": (cube, [[0], [0], [17]]),
            "two sources on a node": (cube, [[5, 5], [6, 6], [7, 7]]),
            "a pml": (replace(sim, pml=Pml(4, 1.0)), [[9], [20]]),
            "a domain": (replace(sim, domain=_random(sim, 1)[0] > 0.5), [[9], [20]]),
            "streaming": (_scalar(6), [[9], [20]]),
            "three nodes": (_scalar(2, Nx=(3, 45)), [[1], [20]]),
            "order 6": (_scalar(6, Nx=(37, 45)), [[9], [20]]),
        }
        for name, (each, points) in kept.items():
            with self.subTest(name=name):
                position = cp.asarray(points, dtype=cp.int32)
                source = Source(position, cp.zeros((1, position.shape[1]), each.dtype))
                mat = each.build_materials(_random(each, 1)[0])
                streams = mock.patch.object(type(each), "streams", name == "streaming")
                with streams, mock.patch.object(scalar, "FOLD_SMS", 0):
                    kernels = compile_kernels(each)
                    self.assertIsNone(each.define_folded_step(kernels, mat, source))


if __name__ == "__main__":
    unittest.main()
