import os
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import fdtdx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# -------------------------------------- settings -------------------------------------
# implementation
OUTPUT = Path(__file__).resolve().parents[2] / "tests" / "reference"

# discretization
SPACING = 1e-7  # metres; fdtdx takes SI, cuwave rescales to dx = 1
COURANT = 0.99  # fdtdx's own default, dt = COURANT / sqrt(3) * dx / c0
POINTS_PER_WAVELENGTH = 20  # in the dense phase

# physics
PERMITTIVITY = 4.0

# cells, steps, block (corner, size) or None, dipole axis, component, refinement
CASES = {
    "tm2D_vacuum": ((96, 80), 400, None, 2, "Ez", 1),
    "tm2D_block": ((96, 80), 400, ((52, 30), (14, 18)), 2, "Ez", 1),
    "te2D_vacuum": ((160, 160), 140, None, 0, "Hz", 1),  # no wall echo returns in time
    "te2D_block": ((160, 160), 140, ((84, 72), (12, 16)), 0, "Hz", 1),
    "te2D_block_fine": ((160, 160), 140, ((84, 72), (12, 16)), 0, "Hz", 2),
    "vector3D_vacuum": ((40, 36, 32), 180, None, 2, "Ez", 1),
    "vector3D_block": ((40, 36, 32), 180, ((22, 12, 10), (8, 10, 12)), 2, "Ez", 1),
}
SOURCE = {"tm2D": (30, 38), "te2D": (76, 80), "vector3D": (14, 18, 16)}
PROBES = {
    "tm2D": [(30, 38), (44, 40), (58, 39), (70, 52), (20, 64), (88, 10)],
    "te2D": [(76, 80), (76, 79), (90, 80), (96, 86), (70, 66), (60, 90)],
    "vector3D": [(14, 18, 16), (20, 18, 16), (26, 16, 15), (31, 22, 20), (8, 30, 6)],
}

# --------------------------------------- setup ---------------------------------------
c0 = 299_792_458.0
courant = COURANT / np.sqrt(3.0)
dt = courant * SPACING / c0
OUTPUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------- helper --------------------------------------
def run(shape, steps, block, polarization, component, source, probes, frequency):
    """One fdtdx run in a PEC box under one electric dipole: (amplitude, traces, steps)."""
    ndim = len(shape)
    volume_shape = (*shape, 2) if ndim == 2 else shape
    config = fdtdx.SimulationConfig(
        time=(steps - 0.5) * dt,
        grid=fdtdx.UniformGrid(spacing=SPACING),
        dtype=jnp.float64,
        courant_factor=COURANT,
        backend="gpu",
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=volume_shape)
    objects, constraints = [volume], []
    # a z-uniform 2D field needs its two z cells periodic, the walls are conductors
    override = {"min_z": "periodic", "max_z": "periodic"} if ndim == 2 else {}
    boundary = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=1, boundary_type="pec", override_types=override
    )
    walls, wall_constraints = fdtdx.boundary_objects_from_config(boundary, volume)
    objects += list(walls.values())
    constraints += wall_constraints

    if block is not None:
        low, size = block
        size = (*size, None) if ndim == 2 else size
        cube = fdtdx.UniformMaterialObject(
            name="block",
            partial_grid_shape=size,
            material=fdtdx.Material(permittivity=PERMITTIVITY),
        )
        constraints.append(
            cube.set_grid_coordinates(
                axes=tuple(range(ndim)), sides=("-",) * ndim, coordinates=low
            )
        )
        if ndim == 2:
            constraints.append(cube.same_size(volume, axes=(2,)))
        objects.append(cube)

    t = np.arange(steps + 2) * dt
    delay = 1.0 / frequency
    arg = (np.pi * frequency * (t - delay)) ** 2
    amplitude = (1.0 - 2.0 * arg) * np.exp(-arg)
    profile = fdtdx.CustomTimeSignalProfile(
        signal=jnp.asarray(amplitude), time_step_duration=dt, interpolation="nearest"
    )
    dipole = fdtdx.PointDipoleSource(
        name="dipole",
        partial_grid_shape=(1, 1, 1) if ndim == 3 else (1, 1, None),
        polarization=polarization,
        wave_character=fdtdx.WaveCharacter(frequency=frequency),
        temporal_profile=profile,
    )
    constraints.append(
        dipole.set_grid_coordinates(
            axes=tuple(range(ndim)), sides=("-",) * ndim, coordinates=source
        )
    )
    if ndim == 2:
        constraints.append(dipole.same_size(volume, axes=(2,)))
    objects.append(dipole)

    for k, probe in enumerate(probes):
        detector = fdtdx.FieldDetector(
            name=f"probe{k}",
            partial_grid_shape=(1, 1, 1),
            components=(component,),
            exact_interpolation=False,
            plot=False,
            dtype=jnp.float64,
        )
        coordinates = (*probe, 0) if ndim == 2 else probe
        constraints.append(
            detector.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-",) * 3, coordinates=coordinates
            )
        )
        objects.append(detector)

    key = jax.random.PRNGKey(0)
    container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, container, _ = fdtdx.apply_params(arrays, container, params, key)
    _, arrays = fdtdx.run_fdtd(
        arrays=arrays, objects=container, config=config, key=key, show_progress=False
    )
    traces = [
        np.asarray(arrays.detector_states[f"probe{k}"]["fields"]).reshape(-1)
        for k in range(len(probes))
    ]
    return amplitude, np.stack(traces, axis=1), config.time_steps_total


# ---------------------------------------- solve --------------------------------------
for name, (shape, steps, block, polarization, component, r) in CASES.items():
    family = name.split("_")[0]
    shape, steps = tuple(r * n for n in shape), r * steps
    if block is not None:
        block = tuple(tuple(r * n for n in corner) for corner in block)
    source = tuple(r * n for n in SOURCE[family])
    probes = [tuple(r * n for n in probe) for probe in PROBES[family]]
    frequency = c0 / (r * POINTS_PER_WAVELENGTH * np.sqrt(PERMITTIVITY) * SPACING)
    amplitude, traces, total = run(
        shape, steps, block, polarization, component, source, probes, frequency
    )
    low, size = block if block is not None else ((), ())
    np.savez(
        OUTPUT / f"fdtdx_{name}.npz",
        shape=np.array(shape),
        courant=courant,
        amplitude=amplitude[:total],
        polarization=polarization,
        source=np.array(source),
        probes=np.array(probes),
        block_low=np.array(low, dtype=int),
        block_size=np.array(size, dtype=int),
        permittivity=PERMITTIVITY,
        traces=traces,
    )
    print(f"{name}: {total} steps, peak {np.abs(traces).max():.3e}")
