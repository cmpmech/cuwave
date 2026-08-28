import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from analog_rnn_setup import (
    DATASET,
    MATERIAL,
    N,
    cross_entropy,
    load_source,
    probabilities,
    region,
    sensors,
    sim,
    to_node,
)

from cuwave.optimization import Adam
from cuwave.postprocessing import show
from cuwave.regularization import DensityFilter, Projection, continuation
from cuwave.sensitivity import reconstruction_sensitivity
from cuwave.utils import threshold
from cuwave.wave import simulate

# -------------------------------------- settings -------------------------------------
# geometry: a trainable material square centered in x, spanning the full height
DESIGN_X = (2.5, 7.5)

# optimization
CLIPS_PER_CLASS = -1  # -1 for every clip the dataset holds
BATCH_SIZE = 1
EPOCHS = 240
BINARY_FROM = 120  # epoch the forward switches to the design that will be built
LR = 2e-2
DESIGN_START = 0.5
DESIGN_NOISE = 0.1  # breaks the symmetry a flat start would leave the probes in
AMPLITUDE_PENALTY = (
    0.7  # rewards energy reaching the probes, so the loss can go negative
)

# regularization
RMIN = 0.2  # metres, so a feature is a scatterer rather than a grey interface
ETA = 0.5
BETA, BETA_MAX, STAGES = 1.0, 128.0, 5
SCHEME = "staircase"  # the ramp has to finish before the binary phase starts

# evaluation
THRESHOLD = 0.5  # the projection maps onto [0, 1], so its midpoint is the cut

# --------------------------------------- setup ---------------------------------------
rng = np.random.default_rng(0)
x_lo, x_hi = to_node((DESIGN_X[0], 0.0))[0], to_node((DESIGN_X[1], 0.0))[0]
design = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
design[x_lo:x_hi, region[1]] = 1.0

density_filter = DensityFilter(RMIN / min(sim.dx), sim.Nx_padded, dtype=sim.dtype)
projection = Projection(BETA, ETA)
betas = continuation(SCHEME, BINARY_FROM, BETA, BETA_MAX, STAGES)
betas += [BETA_MAX] * (EPOCHS - BINARY_FROM)


def physical(variables):
    """Filter then project the design variables: (physical field, filtered field)."""
    filtered = density_filter(variables)
    return projection(filtered) * design, filtered


# ---------------------------------------- data ---------------------------------------
data = np.load(DATASET)
classes, labels, clips = data["classes"], data["y"], data["X"]

if CLIPS_PER_CLASS != -1:
    keep = np.concatenate(
        [np.where(labels == c)[0][:CLIPS_PER_CLASS] for c in range(len(classes))]
    )
    labels, clips = labels[keep], clips[keep]

sources = [load_source(clip) for clip in clips]
samples = len(sources)

# inverse-frequency weights (mean 1) counter an imbalanced clip count per class
counts = np.maximum(np.bincount(labels, minlength=len(classes)), 1)
class_weights = samples / (len(classes) * counts)
print(f"{samples} clips over {len(classes)} classes, {int(design.sum())} design nodes")

# ------------------------------------ optimization -----------------------------------
noise = DESIGN_NOISE * (cp.asarray(rng.random(sim.Nx_padded), dtype=sim.dtype) - 0.5)
variables = (DESIGN_START + noise) * design
optimizer = Adam(lr=LR)
d_mass, d_stiff = sim.parametrization_jacobian()
loss_history, acc_history, margin_history = [], [], []
grad_scale = None

cp.cuda.Stream.null.synchronize()
tic = time.time()
for epoch in range(EPOCHS):
    projection.set(beta=betas[epoch])
    # sharpening scales the projection adjoint by ~beta, so the step size is recaptured
    if epoch == 0 or betas[epoch] != betas[epoch - 1]:
        grad_scale = None
    # the binary phase steps on a different design, so Adam and its schedule restart
    if epoch == BINARY_FROM:
        optimizer, grad_scale = Adam(lr=LR), None
    first = 0 if epoch < BINARY_FROM else BINARY_FROM
    span = BINARY_FROM if epoch < BINARY_FROM else EPOCHS - BINARY_FROM
    optimizer.lr = LR * 0.5 * (1.0 + math.cos(math.pi * (epoch - first) / span))
    order = rng.permutation(samples)

    loss_sum, correct, margins = 0.0, 0, []
    for start in range(0, samples, BATCH_SIZE):
        batch = order[start : start + BATCH_SIZE]
        gamma, filtered = physical(variables)
        # the forward runs on the design that will be built, the adjoint on the grey one
        if epoch >= BINARY_FROM:
            gamma = threshold(gamma, THRESHOLD, dtype=sim.dtype)

        gradient = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
        for i in batch:
            label = int(labels[i])
            objective = cross_entropy(label, AMPLITUDE_PENALTY)
            cost, grads, um, _ = reconstruction_sensitivity(
                sim, sources[i], gamma, sensors.nodes, sensors.objective(objective)
            )
            weight = float(class_weights[label])
            gradient += weight * (d_mass * grads["mass"] + d_stiff * grads["stiff"])
            loss_sum += weight * cost
            probs = probabilities(sensors.traces(um))
            correct += int(probs.argmax() == label)
            margins.append(float(probs[label] - cp.max(cp.delete(probs, label))))
        gradient /= len(batch)

        # chain rule: indicator -> projection -> filter -> variables
        gradient = density_filter.grad(
            variables, projection.grad(filtered, gradient * design)
        )
        # frozen within a beta level, so the step never tracks the loss scale
        if grad_scale is None:
            grad_scale = float(cp.max(cp.abs(gradient)))
        variables = cp.clip(optimizer.step(variables, gradient / grad_scale), 0.0, 1.0)

    loss_history.append(loss_sum / samples)
    acc_history.append(correct / samples)
    margin_history.append(float(np.mean(margins)))
    print(
        f"epoch {epoch}: loss {loss_history[-1]:.4f}  "
        f"accuracy {acc_history[-1]:.2f}  margin {margin_history[-1]:+.3f}  "
        f"beta {projection.beta:.0f}"
    )
cp.cuda.Stream.null.synchronize()
print(f"{EPOCHS} epochs of {samples} clips of {N} steps: {time.time() - tic:.1f}s")

# ------------------------------------- evaluation ------------------------------------
projection.set(beta=BETA_MAX)
gamma, _ = physical(variables)
final = threshold(gamma, THRESHOLD, dtype=sim.dtype)

# the thresholded design is the one that can be built, so it is the one that is reported
confusion = np.zeros((len(classes), len(classes)), dtype=int)
for source, label in zip(sources, labels):
    probs = probabilities(
        sensors.traces(simulate(sim, source, final, sensors.nodes)[1])
    )
    confusion[int(label), int(probs.argmax())] += 1
    print(f"true {classes[int(label)]:8s} probs {np.round(probs.get(), 3)}")
print(f"\nthresholded accuracy {np.trace(confusion) / confusion.sum():.3f}")
print(confusion)

material = (final[region] > THRESHOLD).get()
np.save(MATERIAL, material)
print(f"saved {material.shape} material mask to {MATERIAL.name}")

# ----------------------------------- postprocessing ----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes[0, 0].plot(loss_history, "k")
axes[0, 0].set_title("loss")
axes[0, 1].plot(acc_history, "k", label="accuracy")
axes[0, 1].plot(margin_history, "C3", label="margin")
axes[0, 1].set_ylim(0, 1)
axes[0, 1].legend(loc="lower right")
show(axes[1, 0], indicator=final[region], cmap="binary")
axes[1, 0].set_aspect("equal")
axes[1, 0].axis("off")
axes[1, 1].imshow(confusion, origin="upper", cmap="cividis")
axes[1, 1].set_title("confusion")
fig.tight_layout()
plt.show()
