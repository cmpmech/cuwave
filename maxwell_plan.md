# Maxwell in CuWave — implementation plan

## Discretization choice

Yee's staggered scheme, but in **second-order curl–curl form** rather than the usual
first-order E/H leapfrog:

```
eps d2E/dt2 = -curl( mu^-1 curl E ) - sigma dE/dt
```

Reason: the whole repo — `simulate`, the three-level `(u0, u1, u2)` march, both
sensitivity variants, `stable_timestep`'s power iteration — is built around a symmetric
second-order-in-time operator of the shape `-B^T C B` with a diagonal mass. The
curl–curl form is exactly that with `B = curl`, `C = mu^-1` sampled on the curl points
and `mass = eps` on the field points. The first-order Yee pair would bypass the entire
adjoint machinery; the curl–curl form inherits it (exact transpose, lossless
reversibility, `superposition_sensitivity`) for free.

On the staggered lattice this *is* the Yee lattice: eliminating H from the classical
Yee update recovers the same stencil.

## Where it does / does not make sense

| case | verdict |
|---|---|
| 1D, 2D TM (`Ez, Hx, Hy`) | already covered — reduces to the scalar wave equation, i.e. `scalar.ScalarWave` with `c = 1/sqrt(eps mu)` |
| 2D TE (`Ex, Ey`) | new physics: vector curl–curl with one curl component |
| 3D | new physics: full curl–curl, three E components, three curl components |

So the module implements the 2D TE and 3D cases; TM is a one-line note pointing at
`scalar.ScalarWave` (and doubles as a free cross-check test).

## Mapping onto the existing layout

`MaxwellWave` is structurally `ElasticWave` with the normal-stress block deleted:

- **Staggering.** `E_c` half a node up axis `c` (`component_offsets = 0.5 * eye`),
  identical to the displacement components. The curl component of pair `(k, l)` lands
  half a node up *both* axes — exactly the shear-strain points, so `PAIRS`,
  `pair_weights`, `pair_average` (harmonic mean of `mu^-1` across jumps) and
  `point_average` (arithmetic mean of `eps`) carry over verbatim.
- **Kernels.** `kernels/maxwell.cu`: a `curl_kernel` assembling the `npairs` curl
  components (the shear-strain assembly with the second term's sign flipped:
  `d_k E_l - d_l E_k`) and an `fd_kernel` applying the transposed differences — the
  `stress_kernel` + `fd_kernel` pair of `elastic.cu` minus the diagonal terms. Per-axis
  staggered differences, so any even `space_order` at linear cost, like elastic.
- **Hooks.** `parametrization` (indicator scales `eps`, the photonic design variable;
  `mu` fixed), `parametrization_jacobian`, `step_factors` (`dt^2 / dx`),
  `source_factor`; multi-launch step via `define_step` override, same as elastic.
- **Boundaries.** PEC (tangential `E = 0`) is `Clamped`; PMC is the kernel-less natural
  boundary, i.e. `Traction` renamed/aliased. Sponge absorption reuses `damping`
  (`sigma / eps` is exactly the existing Rayleigh-mass term); a true PML is out of
  scope for the first pass.
- **Sensitivity.** `kernels/maxwell_sensitivity.cu` mirroring
  `elastic_sensitivity.cu`: gradients w.r.t. `(mass, stiff)` = `(eps, mu^-1)` densities
  on the component and pair points, chained onto the nodal design in
  `finalize_gradients` through the same average adjoints. Both `sensitivity` and
  `superposition_sensitivity` work unchanged (the operator is lossless when
  `sigma = 0`).
- **Timestep.** `elastic.stable_timestep` applies as-is (it only needs `define_step`);
  promote it to a shared helper rather than copying it.

One genuine Maxwell-specific point: curl–curl has a null space (discrete gradient
fields). Those modes are stationary, not unstable, and the Yee lattice preserves the
discrete divergence of `eps E` exactly, so a divergence-compatible source (any
transverse current) never excites them. Document this in `docs/maxwell.md`; the dot
test does not care.

## Refactor first: machinery to share instead of copy

Do this as a standalone commit before `maxwell.py`, with the elastic tests green on
both sides of it.

**Python — a thin staggered layer between `Simulation` and the two equations.**
`MaxwellWave` would otherwise copy the bulk of `ElasticWave` verbatim, because most of
it is lattice geometry, not elasticity:

- `stable_timestep` — already generic (it only touches `define_step` and
  `component_offsets`); move it to `wave.py` next to `stable_dt`.
- A `StaggeredWave(Simulation)` base (or `staggered.py` mixin) holding everything that
  is a pure function of `component_offsets` and `PAIRS`: `PAIRS`, `radius`, `reach`,
  `ncomp`/`component_offsets`, `component_weights`, `pair_weights`, `pair_average`,
  `point_average`, `clamped_faces`/`clamped_mask`, `inverse_inertia`, the `minv` loop
  of `build_materials`, `excitation_weights`, `adjoint_weights`, and the two-launch
  `define_step` skeleton (scratch allocation + patch-and-launch closure), parametrized
  by the scratch component count (`nvoigt` vs `npairs`) and the material tail.
- The average adjoints buried in `ElasticWave.finalize_gradients` — the
  arithmetic-mean scatter and the harmonic-mean `(mean/node)^2` scatter — become
  standalone helpers (`point_average_adjoint`, `pair_average_adjoint`) both
  `finalize_gradients` implementations call.

What stays per-equation: the four hooks, the material scalars, `voigt`/plane logic,
and the kernel argument lists. Keep the base thin — the point is deduplicating
lattice geometry, not building a framework.

**CUDA — a shared prelude, prepended like the stencil table.** Every `.cu` currently
repeats ~100 lines: the `real_t`/`USE_FLOAT` header, `STENCIL_RADIUS`/`SG_W`,
`NPAIRS`/`PAIR_ROW`, `rad_node`/`rad_half`, `clamped_face`, the three `AXIS_*` macro
blocks, and the `excitation` / `get_signal` / `set_signal` kernels. Extract them into
`kernels/common.cuh` and have `compile_kernels` prepend it as source, the same
mechanism `stencils.preamble` already uses (so the module cache keys stay correct and
no include paths are needed). This pays off for the three existing equations
immediately and `maxwell.cu` then contains only the curl and update kernels.

Do **not** merge the step kernels themselves: `curl_kernel` is the shear block of
`stress_kernel` with a sign flip, but folding both into one file behind more `#ifdef`s
trades the repo's one-`.cu`-per-equation legibility for ~40 saved lines. Same
judgment for the sensitivity kernels.

## File checklist

0. refactor commit: `StaggeredWave` layer, `stable_timestep` → `wave.py`,
   average adjoints as helpers, `kernels/common.cuh` prelude
1. `cuwave/maxwell.py` — `MaxwellWave(StaggeredWave)` (copy `elastic.py`'s shape)
2. `cuwave/kernels/maxwell.cu`, `cuwave/kernels/maxwell_sensitivity.cu`
3. `tests/maxwell_test.py` — reversibility, adjoint dot test, TM-vs-`scalar.ScalarWave`
   equivalence, plane-wave dispersion vs analytic
4. `examples/forward/maxwell_nD.py` — copy `scalar_nD.py`'s shape
5. `docs/maxwell.md`, `docs/cuda_maxwell.md`, `docs/cuda_maxwell_sensitivity.md`
   + one line each in `docs/Home.md`

Application target: transient photonic inverse design / permittivity FWI — the same
two use cases the repo already serves, with `eps` in the role of the indicator-scaled
material.
