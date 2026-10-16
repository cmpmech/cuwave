from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuwave.boundary import Conductor, Dirichlet, Neumann
from cuwave.maxwell import DielectricWave, ElectricWave, MagneticWave
from cuwave.wave import Source, flatten_indices, simulate

# -------------------------------------- settings -------------------------------------
# implementation
REFERENCE = Path(__file__).resolve().parents[2] / "tests" / "reference"
CASES = (
    "tm2D_vacuum",
    "tm2D_block",
    "te2D_vacuum",
    "te2D_block",
    "te2D_block_fine",
    "vector3D_vacuum",
    "vector3D_block",
)
THREADS = {2: (8, 32), 3: (2, 8, 32)}

# postprocessing
PROBE = 2  # the trace plotted per case

# --------------------------------------- setup ---------------------------------------
# per family: the class, its wall, and the component row a vector unknown is read on
FAMILIES = {
    "tm2D": (ElectricWave, Dirichlet, []),
    "te2D": (MagneticWave, Neumann, []),
    "vector3D": (DielectricWave, Conductor, [2]),
}

# ---------------------------------------- solve --------------------------------------
fig, axes = plt.subplots(len(CASES), 1, figsize=(7, 1.6 * len(CASES)), sharex=True)
for ax, name in zip(axes, CASES):
    ref = np.load(REFERENCE / f"fdtdx_{name}.npz")
    cls, wall, row = FAMILIES[name.split("_")[0]]
    shape = tuple(int(n) for n in ref["shape"])
    ndim = len(shape)
    N = len(ref["amplitude"])
    S = float(ref["courant"])
    # fdtdx cell i is node i + 1 here, so the ghost ring wraps the conductor faces
    sim = cls(
        tuple(n + 2 for n in shape),
        (1.0,) * ndim,
        N,
        S,
        THREADS[ndim],
        precision="float64",
        permittivity1=1.0,
        permittivity2=float(ref["permittivity"]),
        boundary=((wall, wall),) * ndim,
    )
    indicator = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    if ref["block_low"].size:
        low, size = ref["block_low"] + 1, ref["block_size"]
        indicator[tuple(slice(lo, lo + n) for lo, n in zip(low, size))] = 1.0

    # fdtdx adds q = -S a(n dt) to E^(n+1): E's second difference sees q^n - q^(n-1)
    q = -S * ref["amplitude"]
    source = np.array(ref["source"]) + 1
    if cls is MagneticWave:
        # H's sees -S curl q^n, an E_x current curling into the H_z nodes either side
        below = source - np.eye(ndim, dtype=int)[1]
        nodes, loads = [source, below], [-S * q, S * q]
    else:
        nodes, loads = [source], [np.diff(q, prepend=0.0)]
    position = cp.asarray(
        [row * len(nodes)] * len(row) + np.array(nodes).T.tolist(), dtype=cp.int32
    )
    mat = sim.build_materials(indicator)
    weight = cp.asnumpy(sim.excitation_weights(mat, flatten_indices(sim, position)))
    signal = cp.asarray(np.stack(loads, axis=1) / weight, dtype=sim.dtype)

    probes = ref["probes"].T + 1
    probes = cp.asarray(
        [row * probes.shape[1]] * len(row) + probes.tolist(), dtype=cp.int32
    )
    _, traces = simulate(sim, Source(position, signal), indicator, sensors=probes)
    traces = cp.asnumpy(traces)
    reference = ref["traces"]
    worst = np.abs(traces - reference).max() / np.abs(reference).max()
    rel = np.linalg.norm(traces - reference) / np.linalg.norm(reference)
    print(f"{name:16s} max |difference| / peak {worst:.2e}, relative L2 {rel:.2e}")

    ax.plot(reference[:, PROBE], "k", lw=2.5)
    ax.plot(traces[:, PROBE], "C1", lw=1.0)
    ax.set_title(name)

# ----------------------------------- postprocessing ----------------------------------
fig.tight_layout()
plt.show()
