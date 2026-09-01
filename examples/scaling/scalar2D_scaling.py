import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.wave import Source, simulate

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 2  # fixed
PRECISION = "float32"
THREADS = (4, 128)

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
AMPLITUDE = 1e8
CYCLES = 5

RESOLUTIONS = np.logspace(0.7, 4.3, 40).astype(np.int32)  # for laptop (RTX PRO 500)
RESOLUTIONS = np.insert(RESOLUTIONS, 1, RESOLUTIONS[0])

mempool = cp.get_default_memory_pool()
device_total = cp.cuda.Device().mem_info[1]

timings = []
dofs = []
memory = []
for res in RESOLUTIONS:
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

    # --------------------------------------- solve ---------------------------------------
    cp.cuda.Stream.null.synchronize()
    tic = time.time()
    u = simulate(sim, source, indicator, record_every=None)
    cp.cuda.Stream.null.synchronize()
    toc = time.time()

    dofs.append(np.prod(Nx))
    timings.append((toc - tic) / N)
    memory.append(mempool.used_bytes())

    del u, sim, source, indicator
    mempool.free_all_blocks()

    # print(res, dofs[-1])

# ----------------------------------- postprocessing ----------------------------------
dofs_per_s = np.array(dofs) / np.array(timings)
fig, ax = plt.subplots()
ax.plot(dofs[1:], dofs_per_s[1:], "k")
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()

print(
    f"max dofs per s: {dofs_per_s.max() / 1e9:.2f} billion",
    f"@ {dofs[dofs_per_s.argmax()] / 1e6:.2f} million",
)
print(
    f"max dofs: {dofs[-1] / 1e6:.2f} million",
    f"@ {memory[-1] / 1e9:.2f} GB / {device_total / 1e9:.2f} GB",
    f"with {dofs_per_s[-1] / 1e9:.2f} billion dofs/s",
)
