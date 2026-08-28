import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from analog_rnn_setup import (
    DATASET,
    MATERIAL,
    N,
    dt,
    dx,
    load_source,
    probabilities,
    region,
    sensors,
    sim,
    source_coords,
    width,
)

from cuwave.postprocessing import markers, show
from cuwave.wave import simulate

# -------------------------------------- settings -------------------------------------
CLASS = 0  # the first clip of this class is the one played through the medium
SNAPSHOT = 0.5  # seconds into the run the field is drawn at
DESIGN = True  # False plays the same clip through free field, as the reference
SATURATION = 0.2  # fraction of the peak the field colormap runs to

# --------------------------------------- setup ---------------------------------------
if not MATERIAL.exists():
    raise SystemExit(
        f"no trained material at {MATERIAL}; run analog_rnn_train.py first"
    )
material = np.load(MATERIAL) & DESIGN
gamma = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
gamma[region] = cp.asarray(material, dtype=sim.dtype)

data = np.load(DATASET)
clips = np.where(data["y"] == CLASS)[0]
if len(clips) == 0:
    raise SystemExit(f"no clip of class {CLASS} in {DATASET.name}")
source = load_source(data["X"][clips[0]])

# --------------------------------------- solve ---------------------------------------
# every record_every-th step is snapshotted, so frame 1 is the one at SNAPSHOT
step = min(max(int(SNAPSHOT / dt), 1), N - 1)
_, um, frames = simulate(sim, source, gamma, sensors=sensors.nodes, record_every=step)
traces = sensors.traces(um)
probs = probabilities(traces).get()

print(f"clip {data['files'][clips[0]]} ({data['mob'][clips[0]]})")
print({str(name): round(float(p), 3) for name, p in zip(data["classes"], probs)})
print(f"predicted {data['classes'][int(probs.argmax())]}")

# ----------------------------------- postprocessing ----------------------------------
t = np.linspace(0, (N - 1) * dt, N)

fig, axes = plt.subplots(3, 1, figsize=(7, 7), height_ratios=[3, 1, 1])
show(axes[0], field=frames[1][region], indicator=material, saturation=SATURATION)
markers(axes[0], source_coords, dx=dx, origin=width + 1, color="k")
markers(axes[0], sensors.coordinates, dx=dx, origin=width + 1, color="k")
axes[0].set_aspect("equal")
axes[0].axis("off")

axes[1].plot(t, source.signal.sum(axis=1).get(), "k", linewidth=1)
axes[1].set_title("source")
for k, name in enumerate(data["classes"]):
    axes[2].plot(t, traces[:, k].get(), linewidth=1, label=name)
axes[2].set_title("probes")
axes[2].legend(loc="upper left")
for ax in axes[1:]:
    ax.set_xlim(0, t[-1])
    ax.axis("off")
fig.tight_layout()
plt.show()
