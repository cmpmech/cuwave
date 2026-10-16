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
PRECISION = "float32"
THREADS = (1, 8, 32)
DEVICES = (0, 1)  # the split; each resolution also runs on device 0 alone
STEPS = (24, 72)  # two run lengths, whose difference is the per-step cost alone

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
AMPLITUDE = 1e8
CYCLES = 5

RESOLUTIONS = np.arange(256, 1700, 128)  # two RTX 5090

results = {}
for devices in (None, DEVICES):
    dofs, rates = [], []
    for res in RESOLUTIONS:
        seconds = []
        for N in (STEPS[0], *STEPS):  # the first compiles
# --------------------------------------- setup ---------------------------------------
            Nx = (int(res),) * 3
            dx = tuple(LENGTH / (n - 3) for n in Nx)
            dt = 0.95 * min(dx) / WAVESPEED / math.sqrt(3)
            sim = ScalarWave(
                Nx,
                dx,
                N,
                dt,
                THREADS,
                precision=PRECISION,
                wavespeed=WAVESPEED,
                density=DENSITY,
                devices=devices,
            )
            # a split assembles on the host, so the global grid outgrows any one device
            if devices is None:
                indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
            else:
                indicator = np.ones(sim.Nx_padded, dtype=sim.dtype)

# --------------------------------------- source --------------------------------------
            t_np = np.linspace(0, (N - 1) * dt, N)
            signal_np = sineburst(t_np, AMPLITUDE, 20, CYCLES) / np.prod(dx)
            signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)
            source = Source(cp.array([[n // 2] for n in Nx], dtype=cp.int32), signal)

# --------------------------------------- solve ---------------------------------------
            try:
                cp.cuda.Stream.null.synchronize()
                tic = time.time()
                u = simulate(sim, source, indicator)
                cp.cuda.Stream.null.synchronize()
                seconds.append(time.time() - tic)
            except (cp.cuda.memory.OutOfMemoryError, ValueError):
                break
            del u, indicator
            for d in devices or (0,):
                with cp.cuda.Device(d):
                    cp.get_default_memory_pool().free_all_blocks()
        if len(seconds) < len(STEPS) + 1:
            break
        dofs.append(res**3)
        rates.append(res**3 * (STEPS[1] - STEPS[0]) / (seconds[2] - seconds[1]))
    results[1 if devices is None else len(devices)] = (np.array(dofs), np.array(rates))

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
for gpus, (dofs, rates) in results.items():
    ax.plot(dofs, rates, label=f"{gpus} GPU")
    print(
        f"{gpus} GPU: max {dofs[-1] / 1e9:.2f} billion dofs",
        f"at {rates[-1] / 1e9:.1f} billion dofs/s",
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
plt.show()
