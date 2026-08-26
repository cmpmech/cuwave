import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.sensitivity import (
    l2_misfit,
    sensitivity,
    superposition_sensitivity,
)
from cuwave.signals import sineburst
from cuwave.wave import ScalarWave, Source

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 1  # fixed
PRECISION = "float32"
THREADS = (128,)

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
AMPLITUDE = 1e8
CYCLES = 5

# Both variants run one forward and one backward sweep, so ~2x the work of the
# forward scaling driver, and their throughput is close. What separates them is
# memory: the standard adjoint stores the forward history, N + 2 fields, while the
# superposition trick reconstructs it by time reversal and holds 3. At N = 20 that
# measures 27 full grids against 7, so the two upper bounds below differ by about
# 3.9x in node count. Retune them for your device -- and note the standard bound
# shrinks as N grows, while the superposition one does not move at all.
RESOLUTIONS = {
    "standard": np.logspace(1.4, 7.5, 40).astype(np.int32),  # laptop (RTX PRO 500)
    "superposition": np.logspace(1.4, 8.09, 40).astype(np.int32),
}
SUPERPOSITION_SCALE = 1.0  # k; see cuwave/sensitivity.py on how to pick it

mempool = cp.get_default_memory_pool()
device_total = cp.cuda.Device().mem_info[1]

results = {}
for method, resolutions in RESOLUTIONS.items():
    # first resolution repeated, so the run that pays for kernel compilation is
    # dropped from the plot
    resolutions = np.insert(resolutions, 1, resolutions[0])
    timings = []
    dofs = []
    memory = []
    for res in resolutions:
# --------------------------------------- setup ---------------------------------------
        Nx = (res,) * DIM
        dx = tuple(LENGTH / (n - 3) for n in Nx)
        dt = 0.95 * min(dx) / WAVESPEED / math.sqrt(DIM)
        frequency = 20  # bounded by WAVESPEED / (20.0 * min(dx))
        N = 20

        sim = ScalarWave(
            Nx,
            dx,
            N,
            dt,
            THREADS,
            precision=PRECISION,
            wavespeed=WAVESPEED,
            density=DENSITY,
        )
        indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

# --------------------------------------- helper --------------------------------------
        t_np = np.linspace(0, (N - 1) * dt, N)
        signal_np = sineburst(t_np, AMPLITUDE, frequency, CYCLES) / np.prod(dx)
        signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)

        source_pos = cp.array([[1] for n in Nx], dtype=cp.int32)
        source = Source(source_pos, signal)

        # one sensor mid-grid; the misfit against a silent record costs nothing to
        # form and drives an adjoint field of the same size as any objective would
        sensors = cp.array([[n // 2] for n in Nx], dtype=cp.int32)
        objective = l2_misfit(cp.zeros((N, 1), dtype=sim.dtype))

# --------------------------------------- solve ---------------------------------------
        cp.cuda.Stream.null.synchronize()
        tic = time.time()
        if method == "standard":
            cost, grads, traces, _ = sensitivity(
                sim, source, indicator, sensors, objective
            )
        else:
            cost, grads, traces, _ = superposition_sensitivity(
                sim, source, indicator, sensors, objective, scale=SUPERPOSITION_SCALE
            )
        cp.cuda.Stream.null.synchronize()
        toc = time.time()

        dofs.append(np.prod(Nx))
        timings.append((toc - tic) / N)
        # total_bytes, not used_bytes: the history buffer is freed when the call
        # returns, so only the pool's high-water mark still shows what it cost
        memory.append(mempool.total_bytes())

        del grads, traces, sim, source, indicator, sensors, objective
        mempool.free_all_blocks()

    results[method] = (np.array(dofs), np.array(timings), np.array(memory))

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
for method, style in (("standard", "k"), ("superposition", "r")):
    dofs, timings, _ = results[method]
    ax.plot(dofs[1:], dofs[1:] / timings[1:], style, label=method)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
plt.show()

for method in RESOLUTIONS:
    dofs, timings, memory = results[method]
    dofs_per_s = dofs / timings
    print(
        f"{method:>14}: max {dofs_per_s.max() / 1e9:.2f} billion dofs/s"
        f" @ {dofs[dofs_per_s.argmax()] / 1e6:.2f} million"
        f" | ceiling {dofs[-1] / 1e6:.2f} million dofs"
        f" @ {memory[-1] / 1e9:.2f} GB / {device_total / 1e9:.2f} GB"
        f" ({memory[-1] / dofs[-1]:.0f} B/dof)"
    )
