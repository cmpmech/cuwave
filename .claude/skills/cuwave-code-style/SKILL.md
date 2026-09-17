---
name: cuwave-code-style
description: The house style for the cuwave GPU wave solver: annotated signatures, short docstrings with Args/Returns only where they earn it, 87-character banners, cupy idioms, the top-to-bottom driver structure in examples/, unittest conventions, and the CUDA kernel conventions (banner arithmetic, the compile-flag header, macro-vs-device-function, kernel-local naming). Invoke before writing or editing any .py or .cu in this repo, when cleaning a file to match the house style, or when auditing a module. For docs/*.md use cuwave-docs-style instead.
---

# cuwave code style

The guiding value is **short and dense**: the code carries the explanation, comments and
docstrings carry only what the code cannot. Formatting is delegated to black/ruff, so
nothing here is about whitespace that a formatter already decides.

**Ground truth to imitate**, in this order: `cuwave/evals.py` and `cuwave/signals.py` (the
docstring register), `cuwave/wave.py` (the layering and the closure factories),
`cuwave/geometry.py` and `cuwave/utils.py` (validation and shapes), `cuwave/nn.py` (torch),
`examples/forward/scalarND.py` (the minimal driver), `examples/fwi/scalar2D_fwi_adam.py` (the
full driver). When a rule below is unclear, copy what those files do.

For **comment density** specifically the measure is `cuwave/utils.py`,
`cuwave/regularization.py`, `cuwave/geometry.py` and the `examples/fwi/` drivers: not one of
them carries a comment longer than a single line.

## Hard rules

- **Formatting is black/ruff defaults, 88 columns.** Never hand-wrap code to look nicer;
  run the formatter. Docstring summary lines and comments may overrun 88; prefer cutting
  words to wrapping a summary onto a second line.
- **Signatures are annotated** (PEP 484). Types live in the annotation and are **never**
  repeated in the docstring. `-> None` on `__init__` and `__post_init__`.
- **Single backticks** in docstrings: `` `sim` ``, not ``` ``sim`` ```. `docs/` is plain
  markdown, not Sphinx, so the RST double form buys nothing.
- **ASCII only**, everywhere. `2x`, `1e4`, `lambda` and `gamma` spelled out. No greek
  letters, no arrows, no unicode quotes.
- **No em dashes** in docstrings, comments or messages: not `—`, and not the ASCII `--`
  standing in for one. Recast instead: a comma for an aside, a colon before the
  explanation, parentheses for a true parenthetical, a full stop when the two halves stand
  on their own. `--` stays only where it is not punctuation: command-line flags
  (`--book`), banner rules and the `# --- setup ---` driver headings.
- **No aligned `=`.** Single space each side.
- **Every timing brackets `cp.cuda.Stream.null.synchronize()` on both sides.** CuPy
  launches are asynchronous, so a bare `time.time()` pair measures the launch, not the
  kernel.
- **Errors are `raise ValueError(f"...")`** with the offending value interpolated. Message
  lowercase, no trailing period.
- **Optional dependencies stay optional.** `matplotlib` only in `examples/`; `torch` only
  in `cuwave/nn.py` and behind a `skipUnless` in tests. The solver is cupy throughout.

## Imports

Three groups, one blank line between, in order: stdlib, third-party, local. Inside the
package, local imports are relative (`from .wave import Source, simulate`); in `examples/`
and `tests/` they are absolute (`from cuwave.wave import ScalarWave`). No imports after the
first statement, none unused.

Canonical aliases, no others: `import cupy as cp`, `import cupy.typing as cpt`,
`import numpy as np`, `import numpy.typing as npt`, `import cupyx.scipy.ndimage as ndi`,
`import matplotlib.pyplot as plt`, `from torch import nn` (never `import torch.nn as nn`).

## Annotations

Keep the vocabulary fixed rather than inventing per file:

| kind | annotation |
|---|---|
| simulation object | `Simulation`, or the subclass genuinely required (`PressureWave`) |
| device array | `cpt.NDArray`, `cpt.NDArray[cp.int32]`, `cpt.NDArray[cp.bool_]` |
| host array | `npt.NDArray[np.float64]` |
| either module (`resample`) | `cpt.NDArray \| npt.NDArray` |
| caller input passed through `asarray` | `npt.ArrayLike` |
| returned closure | `Callable[[cpt.NDArray], cpt.NDArray]` |
| grid tuples | `tuple[int, ...]`, `tuple[float, ...]` |
| torch | `torch.Tensor`, `nn.Module`, `type[nn.Module]` for a factory |

**Import cycles.** `boundary.py` and `stencils.py` are imported *by* `wave.py`, so they
cannot import `Simulation` at runtime. Annotate them with a deferred reference:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wave import Simulation
```

Annotate the shape-carrying parameters honestly and leave genuinely duck-typed callbacks
loose (`objective`, `f`, `project` in `optimization.py` are plain `Callable`). An
annotation that has to be `Any` is worse than none: drop it and say the shape in `Args:`.

## Docstrings

**One line is the default and stays the majority.** Imperative or noun phrase, parameter
names in single backticks, no period required. This register:

```python
def stable_dt(dx, wavespeed, space_order=2):
    """CFL-stable timestep for an explicit scheme with grid spacing `dx`."""

def padded_shape(Nx):
    """Pad the fastest (last) axis of `Nx` up to a multiple of 32, for coalesced access."""
```

**`Args:` / `Returns:` only where they earn it**: 3+ parameters, or a non-obvious, tuple, or
variable-arity return. Google form, no types, lowercase entries, names matching the
signature exactly. Their job is the **shape, units, and layout** the annotation cannot
express:

```python
def distribute(
    sim: Simulation,
    coords: npt.ArrayLike,
) -> tuple[cpt.NDArray[cp.int32], cpt.NDArray]:
    """Multilinear interpolation of `coords` onto the grid.

    Args:
        sim: the simulation whose grid the coordinates land on.
        coords: (num, ndim) physical coordinates, inside the domain.

    Returns:
        (nodes, weights) of the surrounding 2**ndim cell corners, shaped
        (ndim, num * 2**ndim) and (num, 2**ndim).
    """
```

Earns a block: `simulate`, `sensitivity`, `distribute`, `point_source`, `misfit_gradient`,
`resample`, `stacked_circles`, `random_ellipses`, `Generator.__init__`, the `Regularization`
subclasses. Never: private `_helpers`, one-argument functions, `forward`.

A `Simulation` subclass documents the physics it adds in its class docstring, and each hook
(`parametrization`, `parametrization_jacobian`, `step_factors`, `source_factor`) keeps its
one-liner naming the formula it returns.

## Module docstrings

**Omit by default.** Add one only when the module has a pattern no single function can
state: how its pieces build on one another, or an invariant they all share. Keep it under
about five lines.

Present, and why:

- `wave.py`, the layering: `Simulation` holds grid and compile configuration, the
  equation modules (`scalar.py`, `elastic.py`, `anisotropic.py`, `maxwell.py`) subclass it
  with their material hooks, and the `define_*` factories bind kernels to that
  configuration for `simulate` to loop over.
- `boundary.py`: a condition is a declarative marker that becomes a launch closure once
  the module is compiled, which is why `simulate` needs no preparation phase.
- `regularization.py`: a map that is its own adjoint, so filters compose the way the
  forward/adjoint pair does.
- `geometry.py`: the shared `coords` / `out` accumulation contract.
- `nn.py`: one line plus the paper link.
- test files: which contracts the file pins.

Absent, and correctly so: `evals.py`, `signals.py`, `stencils.py`, `utils.py`,
`optimization.py`. `sensitivity.py`'s long derivation is a deliberate exception for a
subtle algorithm with a knob the caller must set; do not propagate its length.

## Banners

Exactly **87 characters**, lowercase centered label, flush left at column 0 even inside a
block, followed directly by the first `def` (no blank line between). Build as
`"# " + "-" * left + f" {label} " + "-" * right`, where `left + right = 83 - len(label)`
and `left` is the ceiling:

```
# -------------------------------------- helpers --------------------------------------
# -------------------------------- discretization setup -------------------------------
```

Controlled vocabulary, from what the repo already uses. Library: `helpers`, `utilities`,
`general`, `networks`, `discretization setup`, `kernel helpers`, `simulation functions`,
`binary indicator fields`, `forward pass`, `backward pass`, `adjoint excitation`,
`1D` / `2D` / `3D`. Drivers: `settings`, `setup`, `source`, `measurement`, `solve`,
`optimization`, `evaluation`, `postprocessing`, `helper`. One or two file-specific labels
are fine where they earn it (`penalization in objective`, `continuation schedules`).

## Geometry vocabulary

The scheme is nodal, so these blur easily in a comment. Pin them:

| term | meaning |
|---|---|
| **node** | where a value lives. `u`, `stiff`, `minv`, `gamma` are all nodal; `Nx` counts nodes and `dx` is the spacing between them. |
| **ghost node** | index `0` and `Nx[d] - 1` on each axis, outside the domain. Node **1 is the origin**, so the interior runs `1 .. Nx[d] - 2` and the domain spans `(Nx[d] - 3) * dx[d]`. |
| **cell** | the box between adjacent nodes, side `dx`, so its own boundaries lie **on** the nodes. A node owns half a cell on each side, so an interior node owns one cell and a boundary node half, which is exactly `W`, the **cell weight** `apply_cell_weights` applies. |
| **`i +/- 1/2`** | the cell either side of node `i`, named by its midpoint. Each cell carries exactly one stiffness (`gp` at `i` is `gm` at `i + 1`) and one flux (`Dp` at `i` is `Dm` at `i + 1`), so the cell, not the node, is what those quantities belong to. Write *cell stiffness*, *cell flux*, *the two cells the node borders*. |
| **face** | one of the `2 * ndim` outer boundaries of the domain, encoded as `2 * axis + side` and packed into a bitmask for the kernel. `sim.boundary` is indexed by it. **Only** this meaning: never write *cell face*, which would name the nodes. |
| **side** | `0` low or `1` high within one axis: the second half of a face code, and the `(low, high)` pair `canonical_boundary` builds. |

## Naming

`snake_case` functions and variables, `PascalCase` classes, `ALL_CAPS` module constants
(`KERNEL_PATH`, `TOL`, `NAN`, `CONVOLUTIONS`, `OUTPUT_STD`), leading `_` for module-private
helpers.

Reuse these recurring names before inventing one:

| name | meaning |
|---|---|
| `sim` | the `Simulation` instance |
| `Nx`, `Nx_padded` | logical and padded grid points per axis |
| `dx`, `dt` | grid spacing per axis, timestep |
| `N` | number of timesteps |
| `ndim`, `strides` | dimensionality, C-contiguous strides over the padded shape |
| `d` | axis index; `t` timestep index; `i` a generic loop index |
| `u0`, `u1`, `u2` | the field slots of the three-term recursion |
| `um` | the `(N, num_sensors)` sensor record |
| `mat` | the material dict the kernels take |
| `gamma`, `indicator` | the design field |
| `truth`, `observed` | reference field, measured traces |
| `source` / `sources`, `sensors` | one shot or the shot list, the receiver array |
| `cost`, `gradient`, `history` | objective value, its derivative, the per-iteration log |

`define_<thing>(...)` returns a closure named `<thing>_step`.

## Idioms to preserve

These read oddly without their reason, so do not "clean" them:

- **Dataclass config with `__post_init__` derivation.** Non-field class attributes
  (`compile_flags`, `derive_inertia`) are deliberate: they are behaviour
  switches a subclass sets, not constructor arguments. A *field* is the opposite case:
  `boundary` and `damping` are setup a caller states once, so every function taking the
  `Simulation` reads the same operator and no two call sites can disagree about it.
- **A preallocated `args` list inside a closure factory.** `define_step_method` mutates
  `args[0:3]` per call so the hot loop allocates nothing. Never rebuild the list per step.
- **An assigned `lambda` for a one-line driver-local helper**: `surface`, `show_field`,
  `compare`, `to_index`. Permitted in `examples/` (E731 waived), not in `cuwave/`.
- **The docs page, not a comment block, for a derivation.** The cell weights `W` are the
  model: `apply_cell_weights` keeps its one-line comment, and `docs/sensitivity.md` states
  the invariant, why the field is never materialized, and which test pins it.

## Comments

**A comment is one line.** Lowercase, explaining **why**, never restating what a name
already says. If it does not fit on one line then it is not a comment: it is a docstring
line, a module docstring, or a paragraph in `docs/<module>.md`. There is no exception:
not for a derivation, not for a trade-off, not above a group of functions.

Two comment lines in a row are two independent labels (`# physics` then `# air, solid`),
never one sentence wrapped.

```python
# no: restates the call
self.dtype = cp.float32 if self.precision == "float32" else cp.float64  # pick the dtype

# yes: states what the code cannot
# indices 0 and -1 are ghost nodes outside the domain
axes = [(cp.arange(n, dtype=dtype) - 1) * d for n, d in zip(padded_shape(Nx), dx)]

# no: one sentence wrapped over three
# the nodal gradient is an integral over the node's cell and over a sum the
# timestep never weighted, so dt / dx**DIM turns it into the density the levels
# share, without which the comparison measures the units, not the discretization

# yes: names what the quantity is; docs/sensitivity.md derives it
# dt / dx**DIM: nodal integral -> density, so the levels are comparable
```

**Cutting a block is routing it, not deleting it.** Every line that comes out has a
destination:

| what the block was | where it belongs |
|---|---|
| a derivation, or a trade-off the caller has to act on | `docs/<module>.md`, in the closing paragraph |
| an invariant several functions share | the module docstring, under five lines |
| the shape, units or layout of one parameter | that function's `Args:` entry |
| a precondition | the `raise ValueError` that already checks it: the validation *is* the comment |
| a cost the reader would not guess | one trailing line naming the cost |
| a measurement, a benchmark, an investigation log | **nowhere. Delete it.** `docs/` documents functionality, not analyses, and neither does the code |

That last row is the common case and the tempting one: numbers from a run that settled a
question are a record of the investigation, not of the software.

The test is **would a caller act on it**. What a thing is *for*, and when to reach for it
over its sibling, is functionality and stays: `superposition_sensitivity` is *the
memory-efficient alternative to `sensitivity`*, `pr_auc` is the metric that discriminates
*when damage is a small fraction of the grid* where `roc_auc` does not. The knob a caller
has to set stays with it (*aim for a cancellation near 1e4 in float32*). The sweep that
established the knob, the resolution study, the before/after of a bug hunt: all gone.

**The audit.** The rule is mechanical, so check it rather than argue it: no file holds a run
of two or more consecutive prose comment lines, banners and label pairs excepted.

```bash
python3 - <<'EOF'
import glob, re
for f in sorted(glob.glob("cuwave/**/*.py", recursive=True) + glob.glob("tests/*.py")
                + glob.glob("examples/**/*.py", recursive=True)):
    run = 0
    for i, line in enumerate(open(f).read().split("\n")):
        if re.match(r"^\s*#", line) and not re.match(r"^\s*# -{4,}", line):
            run += 1
        else:
            if run >= 2:
                print(f"{f}:{i - run + 1} run of {run}")
            run = 0
EOF
```

Prints are rare and load-bearing: a points-per-wavelength sanity check, the per-iteration
normalized misfit, elapsed time (`print(f"elapsed time {toc - tic:.2f} s")`).

## Drivers (`examples/`)

No `__main__` guard, no module docstring, no argparse: they read top to bottom. Shape,
with the banners abbreviated (they are 87 characters wide, as above):

```
imports
# --- settings ---        ALL_CAPS only, grouped by lowercase sub-comments
# --- setup ---           everything derived from the settings, lowercase
# --- <domain> ---        source / measurement / solve / optimization / evaluation
# --- postprocessing ---
```

`settings` sub-comments come from `# implementation`, `# discretization`, `# physics`,
`# geometry`, `# transducer`, `# defects`, `# optimization`, plus a file-specific one where
it earns it. Anything a setting *derives* (`Nx`, `dx`, `dt`, `sim`) is lowercase and lives
in `setup`: a value fixed by hand is `ALL_CAPS`, a value that is computed is not. The one
exception is the dimension symbols of the naming registry: `N` stays `N` even though it is
computed as `math.ceil(T / dt)`, because it is the same symbol as `sim.N` and the book's
notation, capitalized by convention rather than by being a setting.

A setting carries at most one **trailing** comment, never a block above it. A driver has no
docs page of its own, so anything that will not fit on that line either belongs in the
validation underneath (`raise ValueError(f"every level must divide REFERENCE={REFERENCE}:
{LEVELS}")` says it better than three lines of prose ever did) or does not belong.

Plotting: **a field plot is the bare field and nothing else.** No ticks, no box, no axis
labels, no colorbar, no title. `examples/forward/scalarND.py` is the whole pattern and
the default to copy:

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(u_np.T, cmap="seismic", vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
```

`ax.axis("off")` clears the ticks and the spines together, and the `subplots_adjust` runs
the image to the edge of the figure. Add decoration only when the figure cannot be read
without it, and then only the piece that is missing: `set_title` to tell apart the panels
of a multi-panel figure, a legend where several series share one axis (the scaling
drivers). Those two need `fig.tight_layout()` instead, since `subplots_adjust(top=1)` would
crop them. `imshow(..., origin="lower", extent=[0.0, Lx, 0.0, Ly])` is the alternative to
`pcolormesh` where physical coordinates have to be on the axes, which for a bare field
they do not.

Colormaps: `seismic` for wave fields, `hot` for indicator and damage fields, `binary_r` for
masks, `Spectral_r` for a signed sensitivity.

## Tests

`unittest`, one file per module as `tests/<module>_test.py`. Classes
`XTest(unittest.TestCase)`, methods named as sentences
(`test_the_field_starts_undamaged`, `test_the_latent_is_a_buffer_unless_it_is_learnable`).
Optional dependencies behind `@unittest.skipUnless`. Floating point compared with
`assertAlmostEqual` or an explicit relative tolerance, never `==`.

**The module docstring is where a test file's prose lives**: which contracts it pins, and why
those are the ones that can break. A test body then carries at most one trailing comment:
the method name already says what is asserted, and the tolerance already says how tightly. A
test is not the place to re-derive the thing it checks: the numbers from the run that found
the bug are an investigation log, and the assertion that now guards it is the record.

## CUDA

The `.cu` files are the one place where the code does **not** carry the explanation: there
are no docstrings, so the derivation lives in `docs/cuda_scalar.md` and
`docs/cuda_scalar_sensitivity.md` and the file keeps only labels. Ground truth to imitate:
`cuwave/kernels/scalar.cu`, then `cuwave/kernels/scalar_sensitivity.cu`.

### Formatting

- **clang-format, LLVM defaults: 80 columns, 2-space indent.** Never hand-wrap; the banners
  are the only lines allowed past 80.
- **Trailing comments align the way clang-format aligns them**: one space past the longest
  code line of the contiguous commented group. A label that would push the group past 80 is
  shortened, never wrapped.
- **A section banner is the left half of the 87-character Python banner**: the label with
  its left dashes, trailing dashes cut off:
  `"// " + "-" * ceil((83 - len(label)) / 2) + " " + label`

```c
// ----------------------------- finite difference helpers
// -------------------------------------- kernels
```

- **Between kernels, the full-width rule** `"// " + "-" * 84` (87 characters) and nothing
  else: the kernel's own name is the next line.

### The header block

Every `.cu` opens with the compile-time configuration it responds to, and nothing more:

```c
// Compile-time configuration (set via -D flags from sensitivity.py):
//   USE_FLOAT
//   NDIM = 1 | 2 | 3
//   STENCIL_RADIUS
//   OP_COEFFS
```

List only the flags that file actually uses: `scalar_sensitivity.cu` omits `USE_DAMPING`,
which `require_lossless` rejects before either module is compiled. A new flag is added to
this block and to `Simulation.compile_flags` together.

### Comments

- **No multi-line prose blocks.** The why goes into the matching `docs/cuda_*.md`; the file
  keeps terse trailing labels (`// harmonic mean`, `// inner grad`, `// plus of sc`) and at
  most one line above a stanza inside a kernel
  (`// dJ/dmass: no neighbour and no material load`).
- Name what a quantity *is*, and let the doc derive it: `// d(harmonic mean)/dsc`.
- ASCII only, as everywhere else.

### What the Python side depends on

- Precision is one `real_t` typedef switched on `USE_FLOAT`; kernels never name `float` or
  `double` directly.
- Kernel entry points are named `<thing>_kernel` and live in **one** `extern "C" {` block per
  file, because the name is a string on the Python side
  (`kernels.get_function("fd_kernel")`, `BoundaryCondition.kernel`).
- Pointer parameters carry `const ... *__restrict__`.
- The dimension-dependent argument tail is built with `#if NDIM >= 2 / >= 3` and ordered
  `f0, N0 [, f1, N1, s0] [, f2, N2, s1]`: per-axis factor, logical extent, stride of the
  *previous* axis. `wave.axis_geometry` packs it; `boundary.py` and `sensitivity.py` mirror
  the ordering without the factors.
- `N0, N1, N2` are the logical `sim.Nx` (ghost nodes in, padding out), while the launch is
  sized on `Nx_padded`, so one interior guard retires both the ghost ring and the padding
  tail.
- The fastest axis maps to grid/block `x`, as `grid_block` does on the host, so the innermost
  axis is consecutive threads and the loads coalesce.

### Idioms

- **A macro, not a device function, where the code must declare the caller's variables or
  return early**: `INTERIOR_OR_RETURN`, `AXIS_RADII`, `AXIS_OFFSETS`, `BC_PARAMS`, `BC_GEOM`.
  Everything else is `__device__ __forceinline__`.
- **Compile-time loop bounds, runtime extent as a guard**:
  `for (int k = 2; k <= STENCIL_RADIUS; ++k) if (k <= r)` under `#pragma unroll`, so the loop
  unrolls fully and no thread carries a data-dependent trip count.
- **A `__constant__` table for warp-uniform lookups** (`OP_C`), behind a
  `#if STENCIL_RADIUS == 1` branch that collapses the accessor to a literal, so the generic
  kernel compiles back to the hand-written low-order one.
- **Accumulating kernels use `+=` / `-=` into arrays the caller zeroes**, so a driver may run
  them over a subset of the time steps or with aliased fields.
- **`atomicAdd` wherever two indices may name the same node** (`excitation_kernel`): a plain
  read-modify-write silently loses an update, and a sensor sitting on a source is the normal
  case.
- **A whole `(N, num)` record plus an `offset`, never a per-row view**: slicing a row per step
  costs more host time than the kernel costs on the device.
- **What every `.cu` shares lives in `kernels/common.cuh`**, which `compile_kernels`
  prepends as source next to the stencil table: the `real_t` typedef, the `OP_W` / `SG_W`
  accessors and their `__constant__` tables, `NPAIRS` / `PAIR_ROW`, `rad_node` / `rad_half`,
  `clamped_face`, and the three transfer kernels. A `.cu` opens with the flags **it** still
  responds to, not the prelude's, and holds only its own operator.
- **What only some files share stays duplicated byte-identically**: the `AXIS_*` and
  `INTERIOR_OR_RETURN` blocks differ between equations (the anisotropic adjoint takes no
  per-axis factors), and a strain or curl helper is repeated in the sensitivity file rather
  than hoisted. Each file is still its own `RawModule` from a source string, so keep those
  copies diffable rather than growing the prelude to cover them.

### Naming inside a kernel

| name | meaning |
|---|---|
| `idx` | flat index into the padded array |
| `a0, a1, a2` | per-axis grid indices, `a0` the slowest |
| `s0, s1` | strides of axes 0 and 1 over the padded shape (the last is 1) |
| `o0, o1, o2` | neighbour stride per axis, where only one node either way is needed |
| `r0, r1, r2` | the graded stencil radius per axis, from `CLOSURE` |
| `f0, f1, f2` | per-axis factor including `dt^2`; `F0, F1, F2` the same without it |
| `uc, sc, lc` | the central node's `u`, `stiff`, `lambda`, loaded once |
| `sp, sm` | a material field at the plus / minus neighbour along the axis |
| `gp, gm` | the cell stiffness either side of the node; `dgp, dgm` its derivative |
| `Dp, Dm` | the two cell fluxes of the axis, `Dm` at `i` being `Dp` at `i - 1` |
| `mi` | inverse inertia at the node |
| `ghost, normal` | a ghost node and the signed stride pointing inward |

Documenting a kernel is a separate job with its own conventions; see the
`cuwave-docs-style` skill.
