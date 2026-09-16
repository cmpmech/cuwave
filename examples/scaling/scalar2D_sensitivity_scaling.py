import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.scalar import ScalarWave
from cuwave.sensitivity import (
    l2_misfit,
    sensitivity,
    superposition_sensitivity,
)
from cuwave.signals import sineburst
from cuwave.wave import Source

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 2
PRECISION = "float32"
THREADS = (4, 128)

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
AMPLITUDE = 1e8
CYCLES = 5

RESOLUTIONS = {  # laptop (RTX PRO 500)
    "standard": np.logspace(0.7, 3.75, 40).astype(np.int32),
    "superposition": np.logspace(0.7, 4.04, 40).astype(np.int32),
}
SUPERPOSITION_SCALE = 1.0

mempool = cp.get_default_memory_pool()
device_total = cp.cuda.Device().mem_info[1]

results = {}
for method, resolutions in RESOLUTIONS.items():
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
