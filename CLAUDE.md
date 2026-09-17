# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CuWave is a single-GPU, differentiable finite-difference wave solver. The forward solve is
CuPy + hand-written CUDA (`cp.RawModule`); the adjoint is hand-derived, not autodiff. Torch
appears only in `cuwave/nn.py`, matplotlib only in `cuwave/postprocessing.py` and the
drivers. Applications: full waveform inversion and transient acoustic topology
optimization.

## Environment and commands

The project uses a dedicated virtualenv; the commands below assume it is active.

```bash
python -m pytest tests/ -q          # whole suite, ~20 s on a GPU
python -m pytest tests/sensitivity_test.py -q
python -m pytest \
    "tests/boundary_test.py::DispatchTest::test_mixed_faces_2D" -q
python examples/fwi/scalar2D_fwi_adam.py  # drivers are run directly
```

- Tests are `unittest` classes run under pytest. Everything CUDA is behind a
  `HAS_CUDA` guard and everything torch behind `@unittest.skipUnless`, so a run on a
  machine without a GPU silently skips rather than fails — check the pass/skip counts.
- `ruff` is not a project dependency; run `ruff check` from wherever it is installed
  (88 columns). Do not run `ruff format`: the house-style banners sit at column 0 even
  inside a block, which the formatter would indent. `.cu` files are clang-format LLVM
  defaults, 80 columns.
- Installing torch here must use the cu128 index
  (`pip install torch --index-url https://download.pytorch.org/whl/cu128`); a plain
  `pip install torch` pulls CUDA 13 wheels that break `cupy-cuda12x` at first kernel launch.

## Style is defined by two skills — invoke them

- **`cuwave-code-style`** before writing or editing any `.py` or `.cu`.
- **`cuwave-docs-style`** before writing or editing anything in `docs/`.

They are authoritative and detailed (banner arithmetic, docstring register, import aliases,
kernel conventions). Do not infer the style from a single file instead of reading them.

## Architecture

### The grid contract (crosses every module)

- `Simulation.Nx` is the **logical** extent per axis, **ghost nodes included**. Node 0 and
  node `Nx[d]-1` are ghosts, so the interior is `1 .. Nx[d]-2` and the physical domain
  length is `(Nx[d]-3) * dx[d]`. `grid_coords` puts the origin at node 1.
- `Simulation.Nx_padded` is the **array** shape: the fastest (last) axis rounded up to a
  multiple of 32 for coalesced access. Every device field is allocated at `Nx_padded`;
  kernels are launched over `Nx_padded` but receive `Nx`, so one interior guard retires both
  the ghost ring and the padding tail.
- Kernel arg tails are always ordered `f0, N0 [, f1, N1, s0] [, f2, N2, s1]` — per-axis
  factor, logical extent, stride of the *previous* axis. `wave.axis_geometry` packs it;
  `boundary.py` and `sensitivity.py` reproduce the same ordering by hand.
- Sources and sensors live on **interior** nodes; `sensitivity` raises if they don't
  (`require_interior`), because a ghost node carries no equation and corrupts the gradient.

### Forward solve (`wave.py` framework + one module per equation)

`wave.py` is the framework only: the grid contract, `compile_kernels`, the `define_*`
closure factories and `simulate`. Every equation family is a `Simulation` subclass in its
own module, paired with its own `kernels/<name>.cu` and `kernels/<name>_sensitivity.cu`,
and names them through `kernel_path` / `sensitivity_path` / `default_boundary`. A new
equation is a new module, never an addition to `wave.py`.

Dataclass hierarchy, each level adding one thing:

`Simulation` (grid, dt, precision, `space_order`, `boundary`, `compile_flags`)
→ `scalar.PressureWave` (nodal `stiff`/`minv` fields, optional damping)
→ `ScalarWave` (constant `wavespeed`/`density`, indicator gamma scales both mass and
stiffness — hence `derive_inertia = True`, `minv` never formed) and `AcousticWave`
(two-phase TATO interpolation between `rho1/kappa1` and `rho2/kappa2`).

A subclass supplies four hooks the solver and adjoint both call: `parametrization`,
`parametrization_jacobian`, `step_factors`, `source_factor`. Adding a new wave equation
means adding a subclass with those four, not touching `simulate`. A scheme that needs
several launches per step overrides `Simulation.define_step` instead (the staggered
elastic's stress+update pair does), still without touching `simulate`.

The vector siblings: `elastic.ElasticWave` is staggered (component `c` half a node up
axis `c`, declared via `component_offsets`, which `utils.distribute` honours), carries any
even `space_order` at per-axis cost, and is the exact transpose at every order.
`anisotropic.AnisotropicElasticWave` is the collocated cell assembly: order 2 only (the
gather costs `(2r)**(2*ndim)`), but it takes an arbitrary symmetric Voigt `C`, which the
staggered layout cannot. Both are lossless-reversible with kernel-less Traction/Clamped
boundaries. The staggered scheme's one-step read reach is `2r-1` nodes (`Simulation.reach`),
which sizes the reconstruction strip.

`maxwell.MaxwellWave` is the Yee lattice as `-C^T nu C`, which is the staggered elastic
layout with the normal-stress block deleted, so it shares that whole geometry layer:
`component_weights`, `pair_weights`, `point_average`, `pair_average` and their adjoints are
free functions in `wave.py`, not methods, precisely because two equations need them. It is
2D/3D only and its kernel-less walls are `Conductor`/`Magnetic`. Its class switch
`magnetic` (the `derive_inertia` idiom) adds `-DUSE_MAGNETIC` and the pair-point curl
density; unset, both adjoint kernels carry no stencil at all, since a uniform `mu` leaves
the stiffness with no design dependence.

**2D Maxwell needs none of that.** Out of plane one component survives and the curl-curl
collapses to a flux divergence, so `maxwell.ElectricWave` (E_z, permittivity as inertia)
and `maxwell.MagneticWave` (H_z, inverse permittivity as stiffness) are `PressureWave` on
the existing `scalar.cu`. Both `examples/tpto/` drivers use them. Trap: `MagneticWave` puts
the design in the **stiffness**, which is where the wide-order transpose stops being exact,
so it must run at `space_order = 2`; `ElectricWave` is exact at any order.

**Compile-time configuration is the central idea.** `compile_kernels` builds one
`cp.RawModule` per `(ndim, precision, damping, space_order)` combination: `NDIM`,
`USE_FLOAT`, `USE_DAMPING` and `USE_MAGNETIC` go in as `-D` flags, while two source blocks
are *prepended* so CuPy's module cache keys on them without a flag: the stencil table from
`stencils.preamble(space_order)`, then `kernels/common.cuh`. That prelude holds what every
`.cu` repeated (the `real_t` typedef, the `OP_W`/`SG_W` accessors, `rad_node`/`rad_half`,
`PAIR_ROW`, and the three transfer kernels); the `AXIS_*` macro blocks stay per file, since
the anisotropic adjoint's signature genuinely differs. Any new flag is added to both the
`.cu` header block and `Simulation.compile_flags`.

**Closure factories.** `define_step_method`, `define_boundary`, `define_excitation`,
`define_get_signal`, `define_gradient`, `define_frechet` each build the launch config and a
mutable `args` list once, then return a small closure that patches only the changing pointers
per step. Per-step host work is deliberately near zero — keep it that way when editing the
loops.

`boundary.py` is declarative: `BoundaryCondition` names a kernel string and nothing else, so
`Simulation.boundary` can carry a per-face layout long before compilation; `define_boundary`
groups faces by condition into a bitmask so the default stays one launch.

### Adjoint sensitivities (`sensitivity.py` + `kernels/<equation>_sensitivity.cu`)

Three variants, identical signature and return, differing only in how the forward field is
made available to the backward pass:

| | memory | `sim.damping` | gradient |
|---|---|---|---|
| `sensitivity` | stores all `N + 2` fields | supported | exact transpose of the discretization |
| `reconstruction_sensitivity` | 7 grids and `N + 2` copies of the strip | supported, and what the strip pays for | the same transpose, wherever the strip shields it |
| `superposition_sensitivity` | 5 grids, independent of `N` | rejected (`require_lossless`) | consistent, not exact; `scale` (k) trades a k² bias against round-off |

Both reconstructing variants rebuild the forward field rather than storing it, which is why
`superposition_sensitivity` needs a **lossless operator** and rejects `damping` outright.
Each reports its own health: `info["drift"]` and `info["strip"]` for the strip variant,
`info["cancellation"]` for the superposition one, which past `CANCELLATION_LIMIT` means the
gradient is mostly round-off and the caller must raise `scale`. See `docs/sensitivity.md`.

All three return gradients with respect to `(mass, stiff)` over the padded grid. Converting
to a gradient with respect to the design indicator is the caller's chain rule via
`sim.parametrization_jacobian(indicator)` — `utils.misfit_gradient` is the reference for
how. `source_sensitivity` is the odd one out: it differentiates the same cost with respect
to `source.signal` rather than the material, in four grids and no history.

### Optimization layer

- `optimization.py` — `Adam` and `Lbfgs` (two-loop recursion plus Armijo `search`, which
  takes a forward-only `f(design) -> cost` and an optional `project`). Plain CuPy arrays, no
  framework.
- `regularization.py` — every `Regularization` is a map (`__call__`) carrying its own
  adjoint (`grad`), so they compose like the forward/adjoint pair. **The driver is the only
  place the composition order exists**; each class knows only its own adjoint.
- `nn.py` — torch `Generator` reparametrizing the design field (autograd handles its own
  gradient; only the field-to-cost step is hand-adjointed).
- `utils.py` — the application glue: `distribute`/`point_source`/`shots`/`Sensors`
  (multilinear interpolation onto the grid and its transpose), `interior_slice`/`threshold`,
  the fwi pair `misfit`/`misfit_gradient` and the tato pair `response`/`response_gradient`,
  both routing the chain rule through the private `_reparametrize`. Objectives live here
  too: `energy` (a time integral, tato) and `intensity` (a running DFT, tpto). An objective
  sees the whole `(N, num_sensors)` record, so a spectral figure of merit needs no new
  machinery.
- `evals.py`, `geometry.py`, `signals.py` — scoring, region masks, source wavelets.
  For Maxwell use `ricker`, not `sineburst`: the curl-curl null space is driven by the
  divergence of the source, and only a wavelet with zero mean *and* zero first moment
  leaves no static blob behind.
- `postprocessing.py` — the shared field-figure helpers, in node-index coordinates.

### Drivers (`examples/`)

Flat top-to-bottom scripts, not functions: an ALL-CAPS settings block, then setup →
measurement → optimization → evaluation → postprocessing, separated by 87-character banners.
`matplotlib` reaches a driver through `cuwave.postprocessing`.
`examples/forward/scalarND.py` is the minimal driver and `examples/fwi/scalar2D_fwi_adam.py`
the full one; copy their shape. `examples/tpto/` holds the photonic inverse-design
driver (a metalens), which is `examples/tato/`'s shape with `intensity` in place of
`energy` and a sponge plus `reconstruction_sensitivity`.

### Docs (`docs/`)

An Obsidian wiki (one page per module, indexed by `docs/Home.md`) carrying the *what and
why* — governing equations, symbol-to-identifier bindings, design trade-offs. Adding a module
page and its `Home.md` line is one edit. The `.cu` files have no docstrings by design, so
the `docs/cuda_*.md` pages are the only ones that document arguments.

## Traps

- **dtype is unchecked all the way to the kernel.** `cp.RawModule` takes raw pointers, so a
  float64 `Source.signal` in a float32 sim is read as garbage and only NaNs a few hundred
  steps later. Pin `dtype=sim.dtype` on every array handed to a kernel; note
  `float32_array / np.float64_scalar` promotes.
- **Drivers use nondimensional units** (L = c0 = rho0 = 1). SI-scale fields in float32 give
  gradients ~1e-20, whose square underflows Adam's second moment to zero and yields NaN.
- **Every timing brackets `cp.cuda.Stream.null.synchronize()` on both sides** — launches are
  asynchronous, so a bare `time.time()` pair measures the launch, not the kernel.

