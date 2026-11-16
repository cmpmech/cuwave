"""The browser demo's solver (`web/solver.cpp`) against `simulate`.

The demo compiles cuwave's own scalar kernels, but drives them with its own C++: the
margins behind its faces (outside the domain for Neumann, behind held nodes for
Dirichlet, a sponge for absorbing), the domain cells and tiles of its walls, the
damping, the weights and sine bursts of its excitation and the order of the step, the
excitation and the Neumann refill are ports of `domain`, `dirichlet`, `domain_cells`,
`domain_tiles`, `sponge`, `excitation_weights`, `sineburst` and `simulate`.
`AgreementTest` runs both on one walled, damped grid with every face mode and two
bursts, at orders 2 and 4 and on one thread and four, and holds the demo's field to
`simulate`'s, built natively (in a temporary folder) from the source the browser gets:
bitwise on the CPU backend, which compiles the kernels as the demo does, and to the
GPU's fast-math rounding there. The source schedule a GPU steps with is held to
`excitation_weights` times the signal. It needs a C++ compiler, not a GPU.
"""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cuwave.backend import ON_GPU, cp
from cuwave.boundary import sponge
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.wave import Source, flatten_indices, simulate

WEB = Path(__file__).parent.parent / "web"
HAS_COMPILER = shutil.which(os.environ.get("CXX", "c++")) is not None
NEUMANN, DIRICHLET, ABSORBING = 0, 1, 2


class _Source(ctypes.Structure):
    _fields_ = [("node", ctypes.c_int), ("value", ctypes.c_float)]


@unittest.skipUnless(HAS_COMPILER, "no C++ compiler")
class AgreementTest(unittest.TestCase):
    ORDERS = (2, 4)
    ROWS, COLS, WIDTH = 40, 50, 12  # visible rows and columns, margin nodes
    MODES = (DIRICHLET, NEUMANN, ABSORBING, ABSORBING)  # top, bottom, left, right
    COURANT, DECAY, BETA = 0.5, 2e-3, 0.3
    N = 300
    BURSTS = ((5, 15, 0), (30, 4, 37))  # visible row, column, start step
    AMPLITUDE, FREQUENCY, CYCLES = 1.0, 0.08, 3
    SOURCES = 16  # the schedule's slots a step
    THREADS = (1, 4)  # the pool only grows, so one thread runs first

    @classmethod
    def setUpClass(cls) -> None:
        cls.solvers = {}
        build = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, build)
        for order in cls.ORDERS:
            environment = {
                **os.environ,
                "PYTHON": sys.executable,
                "SPACE_ORDER": str(order),
                "OUT": build,
            }
            subprocess.run(
                ["bash", WEB / "build.sh", "native"], check=True, env=environment
            )
            solver = ctypes.CDLL(str(Path(build) / f"solver{order}.so"))
            solver.field.restype = ctypes.POINTER(ctypes.c_float)
            solver.walls.restype = ctypes.POINTER(ctypes.c_ubyte)
            solver.plan.restype = ctypes.POINTER(_Source)
            cls.solvers[order] = solver

    def _walls(self, shape: tuple[int, ...], offset: int) -> np.ndarray:
        """A wall block between the bursts and a lone wall node, `offset` from the corner."""
        solid = np.zeros(shape, dtype=bool)
        solid[offset + 12 : offset + 30, offset + 28 : offset + 32] = True
        solid[offset + 30, offset + 20] = True
        return solid

    def _prepare(self, order: int) -> ctypes.CDLL:
        """The demo's solver of `order` on the walled grid, the field zeroed."""
        s = self.solvers[order]
        s.configure(
            ctypes.c_float(self.COURANT),
            ctypes.c_float(self.DECAY),
            self.WIDTH,
            ctypes.c_float(self.BETA),
        )
        for face, mode in enumerate(self.MODES):
            s.boundary(face, mode)
        s.resize(self.ROWS, self.COLS, ctypes.c_double(1.0))
        s.clear()
        shape = (s.extent(), s.stride())
        walls = np.ctypeslib.as_array(s.walls(), shape=shape)
        walls[:] = self._walls(shape, self.WIDTH + 1)
        s.rebuild()
        return s

    def _demo(self, order: int) -> np.ndarray:
        """The demo's field after `N` steps, ghosts and margins included."""
        s = self._prepare(order)
        t = 0
        for row, col, start in self.BURSTS:
            s.advance(start - t)
            t = start
            s.burst(
                row,
                col,
                ctypes.c_double(self.AMPLITUDE),
                ctypes.c_double(self.FREQUENCY),
                self.CYCLES,
            )
        s.advance(self.N - t)
        shape = (s.extent(), s.stride())
        return np.ctypeslib.as_array(s.field(), shape=shape).copy()

    def _schedule(self, order: int) -> np.ndarray:
        """The (N, num_bursts) values the demo's plan adds, by burst."""
        s = self._prepare(order)
        values = np.zeros((self.N, len(self.BURSTS)), dtype=np.float32)
        t = 0
        for b, (row, col, start) in enumerate((*self.BURSTS, (0, 0, self.N))):
            plan = s.plan(start - t)
            for k in range(start - t):
                for slot in range(self.SOURCES):
                    entry = plan[k * self.SOURCES + slot]
                    if entry.node >= 0:
                        r, c = divmod(entry.node, s.stride())
                        node = (r - self.WIDTH - 1, c - self.WIDTH - 1)
                        index = [bb[:2] for bb in self.BURSTS].index(node)
                        values[t + k, index] = entry.value
            t = start
            if b < len(self.BURSTS):
                s.burst(
                    row,
                    col,
                    ctypes.c_double(self.AMPLITUDE),
                    ctypes.c_double(self.FREQUENCY),
                    self.CYCLES,
                )
        return values

    def _simulation(self, order: int) -> tuple[ScalarWave, Source]:
        """The same grid for `simulate`: domain, held nodes, sponge and decay, bursts."""
        M = self.WIDTH
        Nx = (self.ROWS + 2 * M + 2, self.COLS + 2 * M + 2)
        sim = ScalarWave(
            Nx=Nx,
            dx=(1.0, 1.0),
            N=self.N,
            dt=self.COURANT,
            threads=(8, 32),
            space_order=order,
            wavespeed=1.0,
            density=1.0,
        )
        interior = np.zeros(sim.Nx_padded, dtype=bool)
        interior[1 : Nx[0] - 1, 1 : Nx[1] - 1] = True
        margins = [
            (slice(1, M + 1), slice(None)),
            (slice(Nx[0] - 1 - M, Nx[0] - 1), slice(None)),
            (slice(None), slice(1, M + 1)),
            (slice(None), slice(Nx[1] - 1 - M, Nx[1] - 1)),
        ]
        stepped = interior & ~self._walls(sim.Nx_padded, M + 1)
        dirichlet = np.zeros_like(interior)
        for margin, mode in zip(margins, self.MODES):
            if mode != ABSORBING:
                stepped[margin] = False
            if mode == DIRICHLET:
                dirichlet[margin] = True
        sim.domain = cp.asarray(stepped)
        sim.dirichlet = cp.asarray(dirichlet & interior & ~stepped)

        indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
        faces = [f for f, mode in enumerate(self.MODES) if mode == ABSORBING]
        layer = sponge(sim, indicator, M, self.BETA, faces=faces)
        sim.damping = layer + sim.dtype(self.DECAY) * cp.asarray(interior, sim.dtype)
        signal = [
            sineburst(
                self.COURANT * (np.arange(self.N) - start),
                self.AMPLITUDE,
                self.FREQUENCY,
                self.CYCLES,
            )
            for _, _, start in self.BURSTS
        ]
        position = [[r + M + 1 for r, _, _ in self.BURSTS]]
        position += [[c + M + 1 for _, c, _ in self.BURSTS]]
        source = Source(
            position=cp.asarray(position, dtype=cp.int32),
            signal=cp.asarray(np.stack(signal, axis=1), dtype=sim.dtype),
        )
        return sim, source

    def test_the_demo_steps_as_simulate(self):
        for order in self.ORDERS:
            sim, source = self._simulation(order)
            indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
            reference = cp.asnumpy(simulate(sim, source, indicator))
            peak = np.abs(reference).max()
            self.assertGreater(peak, 0)
            # the GPU build divides with --use_fast_math (measured 1.3e-5 of the peak)
            tolerance = 1e-4 * peak if ON_GPU else 0.0
            for threads in self.THREADS:
                with self.subTest(order=order, threads=threads):
                    self.assertEqual(self.solvers[order].threads(threads), threads)
                    demo = self._demo(order)
                    np.testing.assert_allclose(demo, reference, rtol=0, atol=tolerance)

    def test_a_coarser_grid_samples_the_old_one(self):
        self._demo(4)
        s = self.solvers[4]
        shape = (s.extent(), s.stride())
        o = self.WIDTH + 1
        field = np.ctypeslib.as_array(s.field(), shape=shape)[o:-o, o:-o].copy()
        walls = np.ctypeslib.as_array(s.walls(), shape=shape)[o:-o, o:-o].copy()
        s.resize(self.ROWS // 2, self.COLS // 2, ctypes.c_double(2.0))
        shape = (s.extent(), s.stride())
        coarse = np.ctypeslib.as_array(s.field(), shape=shape)[o:-o, o:-o]
        solid = np.ctypeslib.as_array(s.walls(), shape=shape)[o:-o, o:-o]
        # the old node under each new node's center, the field zero in a wall
        np.testing.assert_array_equal(solid, walls[1::2, 1::2])
        np.testing.assert_array_equal(coarse, np.where(solid, 0, field[1::2, 1::2]))

    def test_the_plan_adds_the_excitation(self):
        for order in self.ORDERS:
            with self.subTest(order=order):
                sim, source = self._simulation(order)
                indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
                mat = sim.build_materials(indicator)
                lin_index = flatten_indices(sim, source.position)
                weights = sim.excitation_weights(mat, lin_index)
                expected = cp.asnumpy(weights[None, :] * source.signal)
                np.testing.assert_array_equal(self._schedule(order), expected)
