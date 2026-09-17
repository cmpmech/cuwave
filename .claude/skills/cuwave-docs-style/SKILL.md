---
name: cuwave-docs-style
description: The house style for the cuwave wiki under docs/: the bold-lead opening, the three-column member table that carries signatures without documenting arguments, display math bound to code identifiers by the "with ..." sentence, author-year DOI links, the closing trade-off paragraph, and the separate aim/args/how register the CUDA pages need because .cu files have no docstrings. Invoke before writing or editing any file in docs/, when filling in a TODO stub, or when auditing a page against the role models.
---

# cuwave docs style

`docs/` is a wiki of plain markdown, read in a renderer that does LaTeX, not Sphinx and not
docstrings. Its job is the **what and why**: the governing equation, the symbol-to-identifier
map, the design consequence. The **how** is the code, and the per-argument detail is the
docstring.

**Ground truth to imitate**, in this order: `docs/regularization.md` and
`docs/optimization.md` (the class-family register), `docs/wave.md` (a module whose pieces
build on one another), `docs/cuda_scalar.md` (the kernel register), `docs/sensitivity.md` (a
module that is two alternatives, and the page that says when to pick which),
`docs/Home.md` (the index).

## Hard rules

- **A page documents functionality, not an analysis.** What the module is for, what each
  member does, when to reach for one over its sibling, the governing equation, the knob a
  caller has to set. **Not** a benchmark, a resolution study, a convergence sweep, or the
  before/after of a bug hunt: those are a record of an investigation and belong in neither
  the page nor the code. The test is whether a caller would act on it:
  `superposition_sensitivity` *is the memory-efficient alternative to `sensitivity`* and
  `pr_auc` *discriminates where `roc_auc` does not once damage is a small fraction of the
  grid*, both functional, both stay. The 40 -> 80 -> 160 refinement table that proved it
  does not.
- **No em dashes.** Neither `—` nor the code's ASCII `--`. Recast the sentence instead: a
  comma for an aside, a colon before the explanation or the list it introduces, parentheses
  for a true parenthetical, a full stop when the two halves stand on their own. This holds
  for every page, table cell and figure caption.
- **Never document a function's arguments on a Python page.** The annotated signature and the
  docstring already carry them. A page names the callable *with its defaults* and says what
  it returns and why, and nothing per-parameter.
- **The CUDA pages are the exception**, and the only one: a `.cu` has no docstring, so
  `docs/cuda_*.md` documents every argument. See *CUDA pages* below.
- **Every symbol is bound to its identifier once**, in a `with ...` sentence right under the
  equation, with the backticked name immediately followed by its math symbol, in that order:

  > with the learning rate `lr` $\eta$, the exponential decay rates `betas`
  > $(\beta_1,\beta_2)$ and the guard `eps` $\epsilon$ against division by zero
- **`\textrm{}` for words inside math**, never `\text{}` or `\mathrm{}`:
  `x_\textrm{min}`, `\textrm{order }1\textrm{ (smoothing):}`.
- **Single backticks** for every identifier, filename and flag. No double backticks.
- **No trailing period** at the end of a paragraph, bullet or table cell. Sentences *inside* a
  paragraph are punctuated normally.
- **One page per module**, named after it, linked from `docs/Home.md` under its phase. Adding
  a page and adding its `Home.md` line are one edit.
- Unicode is allowed here (unlike in code): `$\gamma$` and friends in math. Do not carry the
  code's ASCII-only rule into `docs/`, and do not spend the freedom on an em dash either.
- **`face` means a domain boundary and nothing else.** The quantity at $i\pm\tfrac{1}{2}$
  belongs to the **cell** either side of the node: *cell stiffness*, *cell flux*,
  *cell-flux stencil*, *the two cells the node borders*. Never *cell face*: a cell's own
  boundaries lie on the nodes, so it names the wrong thing. The geometry table in
  `cuwave-code-style` pins the whole vocabulary.

## Page shape

```
# Title                    the module, capitalized: "# Regularization", "# CUDA kernels"
<bold-lead opening>        one sentence saying what the module is for
<shared interface>         the table, or the numbered recipe
## Class / phase           one per public class, mirroring the inheritance tree
### Subclass
#### Sub-subclass
```

**The opening** is one sentence with the subject in bold, defining the module in the
vocabulary of the field:

> **Regularization** infuses prior knowledge about the design variables `x` into the
> optimization to make the problem less ill-posed

For a module that is a *procedure* rather than a family, the opening is a numbered recipe
instead (`docs/wave.md`: "a wave simulation is set up as follows: 1. ... 5. run simulation
with `simulate`"). Either way it comes before any heading.

**A class family** puts the shared interface table in the base-class section and gives each
subclass only what it overrides, as a two-column table:

```
| member | specific to `ScalarWave` |
|---|---|
| `step_factors()` | $2\,c_0^2\Delta t^2/\Delta x_k^2$, carrying the wave speed |
```

## The member table

The one device that carries signatures. Three columns, lowercase role label, callable with
its defaults, one sentence:

```
| member | signature | description |
|---|---|---|
| constructor | `Adam(lr=1e-2, betas=(0.9, 0.999), eps=1e-8)` | stores the step size and the two decay rates, initializing both moments and the counter to zero |
| update | `step(x, grad)` | advances the internal state and returns the new design, leaving `x` untouched |
```

Role labels come from what the member *is for*, not from its name. In use: `constructor`,
`update`, `derivative`, `map or penalty`, `penalty`, `filter`, `projection`, `penalization`,
`line search`, `alternative`, `materials`, `coefficients`, `per-axis factors`,
`source scaling`, `kernel arguments`, `kernel options`, `excitation weights`,
`derived layout`.

Other table shapes are free where the content is genuinely tabular: `| flag | set from |
effect |`, `| scheme | schedule | character |`, `| factory | module | kernel | closure |`,
`| convention | reason |`. Prefer a table over a bullet list whenever every entry answers the
same two or three questions.

## Math

- `$$...$$` on its own line, directly after the sentence that introduces it, no blank line
  and no "where" preamble.
- Several related equations share one block, separated by `\qquad`. Cases get a
  `\textrm{label:}\qquad` prefix.
- `\left( ... \right)` around anything containing a fraction or a derivative.
- The equation states the definition; the following `with ...` sentence does the binding and
  adds the one condition that matters ("where the moments start from zero and are therefore
  bias-corrected by $1-\beta_i^t$").
- Notation follows the code's own symbols: $\gamma$ the design field, $m$ inertia, $k$
  stiffness, $d$ damping, $u$ the field, $\lambda$ the adjoint field, $h$ or $\Delta x$ the
  spacing, $\Delta t$ the step, $N$ the step count, $C$ or $J$ the objective.

## Citations

Inline markdown, author-year, DOI link, woven into the sentence with `following`,
`going back to`, `in the form given by` or `see ... on`:

> following [Bruns & Tortorelli 2001](https://doi.org/10.1016/S0045-7825(00)00278-4)
> ([Nocedal & Wright 2006](https://doi.org/10.1007/978-0-387-40065-5), eq. 7.20)

`&` between two authors, three spelled out (`Wang, Lazarov & Sigmund 2011`). A method with a
paper gets the paper; a repo-specific choice does not get a fake one.

## The closing paragraph

**Almost every section ends with one unlinked paragraph the equations do not show**: a
trade-off, a caveat, or what is left to the caller. This is the part a reader cannot
reconstruct from the code, so it is the part worth writing.

> Small `eps` approaches the exact total variation but stiffens the gradient, so it trades
> interface sharpness against the step size the optimizer tolerates

> Neither optimizer enforces box constraints, so clipping the returned design to e.g. $[0,1]$
> is left to the driver

> The weight sums are evaluated with zero padding, so cells at the boundary are normalized by
> their truncated neighborhood only

Reach for `trades X against Y`, `is left to the driver`, `so ... only`, `the price is`.
State the trade-off as a rule the caller can act on, never as the measurement that
established it: *aim for a cancellation near 1e4 in float32*, not the sweep of `k` behind it.

## Cross-references

`see [boundary](boundary.md)`, and that is the whole section when the content lives
elsewhere. `docs/wave.md` ends with two such one-line sections rather than repeating
`define_boundary` and `define_gradient`. Weave a reference into the sentence where it is
natural: "the pair the [sensitivity](sensitivity.md) analysis contracts the adjoint with".

## CUDA pages

`docs/cuda_scalar.md` and `docs/cuda_scalar_sensitivity.md` carry a register of their own,
because a `.cu` has no docstrings: **here the arguments must be documented, and only here**.

Order: `## compilation logic` -> the helpers (macros, then `__device__` functions) ->
`## kernels`. Each helper and kernel is a `###` with bold inline labels on their own line and
the content on the next:

```
### flux_divergence_axis
**aim**
compute $\nabla\cdot(k\nabla u)$ at $i$  (`idx`)
**input args**
`u1`: array of $u$; `stiff`: array of $k$; `idx`: index; `s`: stride along axis
**internal args**
`sp, sm`: `stiff` at the two neighbours $k_{i\pm1}$; `gp, gm`: cell stiffnesses
**how?**
- approximation of outer gradient (flux divergence)
$$\nabla\cdot(k\nabla u)|_i\approx\ldots$$
```

- Labels, in this order where present: `**parallelization**`, `**aim**`, `**input args**`,
  `**internal args**`, `**output args**`, `**how?**`.
- Args are one semicolon-separated run, `` `name`: what it is ``. Give the *shape and units*
  the C signature cannot: which time level a field holds, what a factor has folded into it,
  whether an extent counts ghost nodes.
- `**how?**` is a bulleted list that mixes prose bullets and `$$` blocks freely.
- A kernel identical to its neighbour says so in one line and lists only the difference
  (`homogeneous_dirichlet_kernel`: "Same parallelization, input args and grouping as
  `homogeneous_neumann_kernel`, with the odd mirror in place of the even one").
- Flag the traps a reader will otherwise walk into: two arguments with the same name and
  different content across files, a name that follows a partner rather than its own time
  index.
- The second page **cross-references the first** rather than repeating it: no compilation
  table, no `OP_W` / `CLOSURE` entry.

## What the code sends here

`cuwave-code-style` holds every comment to a single line, so a derivation or a caller-facing
trade-off that outgrows one line arrives here. It lands in the closing paragraph of the
section it belongs to, bound to identifiers by the usual `with ...` sentence, not as a
transplanted comment block and not as a section of its own.

What arrives with it and must be dropped on the way in: the measurements, the sweeps, the
"measured on a 40x36 grid" asides, the account of how a bug was found. A page that reads as
the log of an investigation has taken the wrong half.

## Not yet implemented

A module the repo does not have yet still gets its page and its `Home.md` line, in the shape
`docs/wave.md` uses for `ElasticWave`: the heading, one sentence naming what it will be, and
`**TODO** not implemented yet`, so the index never links to nothing. The repo currently
carries no such stub: every page in `docs/` documents a module that exists.
