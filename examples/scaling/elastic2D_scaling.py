import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.elastic import ElasticWave
from cuwave.signals import sineburst
from cuwave.utils import point_source
from cuwave.wave import simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 2  # fixed
PRECISION = "float32"
SPACE_ORDER = 2
THREADS = (8, 32)

# physics
LENGTH = 1
WAVESPEED_P = 1.0
WAVESPEED_S = 0.5
DENSITY = 1
PLANE = "strain"
AMPLITUDE = 1e4
CYCLES = 5

RESOLUTIONS = np.logspace(0.7, 3.98, 40).astype(np.int32)  # for laptop (RTX PRO 500)
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
        plane=PLANE,
    )
    indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

    # --------------------------------------- helper --------------------------------------
    t_np = np.linspace(0, (N - 1) * dt, N)
    signal = sineburst(t_np, AMPLITUDE, frequency, CYCLES)

    # a point force along the last axis, so both wave types are excited
    centre = [0.5 * LENGTH] * DIM
    direction = [0.0] * (DIM - 1) + [1.0]
    source = point_source(sim, [centre], signal, direction=direction)

    # --------------------------------------- solve ---------------------------------------
    cp.cuda.Stream.null.synchronize()
    tic = time.time()
    u = simulate(sim, source, indicator, record_every=None)
    cp.cuda.Stream.null.synchronize()
    toc = time.time()

    dofs.append(sim.ncomp * np.prod(Nx))  # one displacement component per axis
    timings.append((toc - tic) / N)
    memory.append(mempool.used_bytes())

    del u, sim, source, indicator
    mempool.free_all_blocks()

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
