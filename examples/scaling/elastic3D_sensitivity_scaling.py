import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.elastic import ElasticWave
from cuwave.sensitivity import (
    l2_misfit,
    sensitivity,
    superposition_sensitivity,
)
from cuwave.signals import sineburst
from cuwave.utils import Sensors, point_source
from cuwave.wave import stable_dt

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 3
PRECISION = "float32"
SPACE_ORDER = 2
THREADS = (2, 8, 32)

# physics
LENGTH = 1
WAVESPEED_P = 1.0
WAVESPEED_S = 0.5
DENSITY = 1
AMPLITUDE = 1e4
CYCLES = 5

RESOLUTIONS = {  # laptop (RTX PRO 500)
    "standard": np.logspace(0.7, 2.35, 40).astype(np.int32),
    "superposition": np.logspace(0.7, 2.50, 40).astype(np.int32),
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
        dt = 0.95 * stable_dt(dx, WAVESPEED_P, SPACE_ORDER)
        frequency = 20  # bounded by WAVESPEED_S / (20.0 * min(dx))
        N = 20

        sim = ElasticWave(
            Nx,
            dx,
            N,
            dt,
            THREADS,
            precision=PRECISION,
            space_order=SPACE_ORDER,
            density=DENSITY,
            wavespeed_p=WAVESPEED_P,
            wavespeed_s=WAVESPEED_S,
        )
        indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

# --------------------------------------- helper --------------------------------------
        t_np = np.linspace(0, (N - 1) * dt, N)
        signal = sineburst(t_np, AMPLITUDE, frequency, CYCLES)

        centre = [0.5 * LENGTH] * DIM
        direction = [0.0] * (DIM - 1) + [1.0]
        source = point_source(sim, [centre], signal, direction=direction)

        sensors = Sensors(sim, [[0.25 * LENGTH] * DIM], direction=direction)
        objective = sensors.objective(l2_misfit(cp.zeros((N, 1), dtype=sim.dtype)))

# --------------------------------------- solve ---------------------------------------
        cp.cuda.Stream.null.synchronize()
        tic = time.time()
        if method == "standard":
            cost, grads, traces, _ = sensitivity(
                sim, source, indicator, sensors.nodes, objective
            )
        else:
            cost, grads, traces, _ = superposition_sensitivity(
                sim,
                source,
                indicator,
                sensors.nodes,
                objective,
                scale=SUPERPOSITION_SCALE,
            )
        cp.cuda.Stream.null.synchronize()
        toc = time.time()

        dofs.append(sim.ncomp * np.prod(Nx))
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
