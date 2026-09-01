import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.evals import l2_error
from cuwave.optimization import Adam
from cuwave.scalar import ScalarWave
from cuwave.sensitivity import l2_misfit, source_sensitivity
from cuwave.signals import sineburst
from cuwave.utils import Sensors, collect_source, line, point_source, resample
from cuwave.wave import simulate, stable_dt

# -------------------------------------- settings -------------------------------------
# discretization
SPACE_ORDER = 4
SPACE_ORDER_OBS = 8  # the measurement is simulated more accurately than it is inverted
PRECISION = "float32"
RESOLUTION = (256, 256)
SAFETY = 0.9

# physics
T = 2.5  # traversals per length (y)
WAVESPEED, DENSITY = 1.0, 1.0  # material, homogeneous and known
AMPLITUDE, FREQUENCY, CYCLES = 1.0, 20.0, 4  # the signal to be recovered
SIGNAL_WINDOW = 0.3  # the transducer is known to emit within it, the burst needs 0.1

# geometry
LENGTHS = (1.0, 1.0)

# transducer
SOURCE = (0.5, 1.0)  # emitting into the domain from the top surface
NUM_SENSORS = 32
ARRAY_SPAN = (0.1, 0.9)  # absolute values

# optimization
ITERS, LR = 100, 5e-2

# --------------------------------------- setup ---------------------------------------
Nx = RESOLUTION
dx = tuple(LENGTHS[d] / (Nx[d] - 3) for d in range(len(Nx)))

N = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER))) + 1
N_obs = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER_OBS))) + 1
dt, dt_obs = T / (N - 1), T / (N_obs - 1)

print(f"{WAVESPEED / (FREQUENCY * max(dx)):.0f} points per wavelength, {N} steps")

scalar_wave = lambda N, dt, space_order: ScalarWave(
    Nx,
    dx,
    N,
    dt,
    (4, 64),
    precision=PRECISION,
    space_order=space_order,
    wavespeed=WAVESPEED,
    density=DENSITY,
)
sim = scalar_wave(N, dt, SPACE_ORDER)
sim_obs = scalar_wave(N_obs, dt_obs, SPACE_ORDER_OBS)

# the medium is known, so it is a constant of the inversion rather than its design
indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

# ------------------------------------ measurement ------------------------------------
burst = lambda t: sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)
t = np.linspace(0, T, N)
truth = cp.asarray(burst(t), dtype=sim.dtype)
sensor_coords = line((ARRAY_SPAN[0], 0.0), (ARRAY_SPAN[1], 0.0), NUM_SENSORS)
sensors = Sensors(sim, sensor_coords)
sensors_obs = Sensors(sim_obs, sensor_coords)

record = lambda sim, sensors, source: sensors.traces(
    simulate(sim, source, indicator, sensors=sensors.nodes)[1]
)

cp.cuda.Stream.null.synchronize()
tic = time.time()
truth_obs = cp.asarray(burst(np.linspace(0, T, N_obs)), dtype=sim_obs.dtype)
observed = resample(
    record(sim_obs, sensors_obs, point_source(sim_obs, SOURCE, truth_obs)),
    dt_obs,
    dt,
    N,
)
cp.cuda.Stream.null.synchronize()
print(
    f"one shot of {N_obs} steps at space order {SPACE_ORDER_OBS} "
    f"recorded at {NUM_SENSORS} receivers: {time.time() - tic:.1f}s"
)

# ------------------------------------ optimization -----------------------------------
objective = sensors.objective(l2_misfit(observed))
signal = cp.zeros(N, dtype=sim.dtype)  # the zero-source start, no prior on the shape
# outside the window Adam would fit reverberation: it steps on the sign, not the size
window = cp.asarray(t <= SIGNAL_WINDOW, dtype=sim.dtype)
optimizer = Adam(lr=LR)
history = []

cp.cuda.Stream.null.synchronize()
tic = time.time()
for iteration in range(ITERS):
    source = point_source(sim, SOURCE, signal)
    cost, columns, _, _ = source_sensitivity(
        sim, source, indicator, sensors.nodes, objective
    )
    gradient = window * collect_source(sim, SOURCE, columns)[:, 0]
    # no clip: a signal is unbounded, where a material indicator is not
    signal = optimizer.step(signal, gradient)

    history.append(cost)
    print(f"{iteration}/{ITERS}: normalized misfit {cost / history[0]:.4e}")
# final misfit, so the curve ends at the converged signal rather than one step before
final = record(sim, sensors, point_source(sim, SOURCE, signal))
history.append(l2_misfit(observed)(final)[0])
cp.cuda.Stream.null.synchronize()
print(f"{ITERS}/{ITERS}: normalized misfit {history[-1] / history[0]:.4e}")
print(f"elapsed time: {time.time() - tic:.1f}s")

# ------------------------------------- evaluation ------------------------------------
print(f"\nrelative L2 error of the recovered signal: {l2_error(signal, truth):.4f}")

# ----------------------------------- postprocessing ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].semilogy(np.array(history) / history[0], "k")
axes[0].set_title("normalized misfit")
axes[1].plot(t, truth.get(), "k", label="truth")
axes[1].plot(t, signal.get(), "r--", label="recovered")
axes[1].set_xlim(0.0, SIGNAL_WINDOW)  # the window the design lives in
axes[1].set_title("source signal")
axes[1].legend()
fig.tight_layout()
plt.show()
