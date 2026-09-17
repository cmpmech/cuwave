# Wave
a wave simulation is set up as follows:
1. define wave equation (& parametrization) with a `Simulation` subclass
	- `PressureWave`, with `ScalarWave` and `AcousticWave`
	- `ElasticWave`
	- `AnisotropicElasticWave`
2. define source with `Source`
3. define `indicator` (if heterogeneous)
4. define postprocessing (which signals to save)
5. run simulation with `simulate`
## simulate
`simulate` has two jobs:
1. preparation of the kernels
	- `fd_step` (spatial discretization)
	- `bc_step` (modification for boundary conditions)
	- `excitation_step` (source contribution)
2. unrolling of the time integration (where kernels are executed at each time step)

- the `Simulation`class (and its derived versions) collect the simulation setup
- the `define_[kernel]` prepare the specific kernels
## Simulation
`Simulation` is the base dataclass holding the **discretization** & **implementation** settings

From those it derives everything the kernels are launched and compiled with: so a `Simulation` is built once per setup and then passed around read-only

| member         | signature         | description                                                                                                                                                 |
| -------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| derived layout | `__post_init__()` | resolves the dimension `ndim`, the padded shape `Nx_padded`, the C-contiguous `strides` and the `dtype`, and rejects a `space_order` that is odd or below 2 |
| kernel options | `compile_flags`   | extra `nvcc -D` flags, empty here and extended by the derived classes                                                                                       |
| step | `define_step(kernels, mat)` | the step closure, one `fd_kernel` launch here; a scheme needing several launches per step overrides it, which is how the staggered [elastic](elastic.md) exists without touching `simulate` |
| staggering | `component_offsets` | `(ncomp, ndim)` grid offsets of each component, `None` for a nodal unknown; what [distribute](utils.md) shifts by |
| strip radius | `reach` | nodes one step reads past a point, `space_order // 2` here; what sizes the [reconstruction](sensitivity.md) strip |
| measured timestep | `stable_timestep(sim, indicator, iterations=60, safety=0.95)` | the largest stable step, power-iterated on the step kernel rather than estimated from the speed and the spacing as `stable_dt` is |

A specialization adds the physics: it turns one design field `indicator` $\gamma$ into the nodal coefficient fields the kernels read, and supplies the constants that fold $\Delta t$, $\Delta x$ and the material scaling into the update

| member             | signature                                  | description                                                                                                                                                    |
| ------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| materials          | `build_materials(indicator)` | evaluates the parametrization on $\gamma$ and returns the nodal fields the kernels read, `damping` among them, mirroring their ghost ring into the value the reflecting wall implies |
| coefficients       | `parametrization(indicator)`               | maps $\gamma$ onto the coefficient fields, returning `None` for the one the kernel derives                                                                     |
| derivative         | `parametrization_jacobian(indicator)`               | the pair $\left(\partial m/\partial\gamma,\,\partial k/\partial\gamma\right)$ the [sensitivity](sensitivity.md) analysis contracts the adjoint with            |
| per-axis factors   | `step_factors()`                           | the constant multiplying the flux difference along each axis                                                                                                   |
| source scaling     | `source_factor()`                          | the constant the excitation carries on top of $\Delta t^2/m$                                                                                                   |
| kernel arguments   | `step_kernel_args(mat)`                    | packs the material fields into the argument tuple of the step kernel                                                                                           |
| excitation weights | `excitation_weights(mat, lin_index)`       | evaluates $\Delta t^2/m$ at the source nodes only, so a derived inertia field is never formed over the whole grid, carrying the damped update's divisor $1/\left(1+\beta\right)$ with $\beta=d\,\Delta t/2m$ so the injected impulse does not depend on $d$                                              |
| kernel options     | `compile_flags`                            | adds `-DUSE_DAMPING` once `damping` is set                                                                                          |
## wave equations
Each equation family lives in its own module and fills the hooks above, so adding one means adding a subclass, not touching `simulate`

| family | module | unknown | what it buys |
|---|---|---|---|
| `PressureWave` with `ScalarWave` and `AcousticWave` | [scalar](scalar.md) | one per node | the cheapest operator, any even `space_order`, and the two-phase interpolation the [tato](tato.md) driver optimizes over |
| `ElasticWave` | [elastic](elastic.md) | `ndim` per node, staggered | the vector equation at a cost linear in the stencil radius, and the exact transpose at every order |
| `AnisotropicElasticWave` | [anisotropic](anisotropic.md) | `ndim` per node, collocated | an arbitrary symmetric Voigt $\mathbf{C}$, at order 2 only |
| `MaxwellWave` with `DielectricWave` | [maxwell](maxwell.md) | `ndim` per node, staggered | the curl-curl equation on the Yee lattice, which is the elastic layout with the normal-stress block deleted |
| `PolarizedWave` with `ElectricWave` and `MagneticWave` | [maxwell](maxwell.md) | one per node | the 2D out-of-plane reductions, which are `PressureWave` with its two coefficient fields exchanged |

## define_\[kernel]
Every kernel launch in the time loop is prepared by a `define_[kernel]` factory: it is called **once** during setup and returns a closure that launches one compiled kernel

Everything that does not change between time steps is resolved at that point: the launch configuration, the flat index arrays, the material and geometry arguments, and the scalar casts

```python
args = [None, None, None, *sim.step_kernel_args(mat), *axis_geometry(...)]

def fd_step(u0, u1, u2):
    args[0], args[1], args[2] = u0, u1, u2
    fd_kernel(grid, block, args)
    return u2
```

### kernels

| factory              | module           | kernel              | closure                                          |
| -------------------- | ---------------- | ------------------- | ------------------------------------------------ |
| `define_step_method` | `wave.py`        | `fd_kernel`         | `fd_step(u0, u1, u2)`                            |
| `define_boundary`    | `boundary.py`    | one per condition   | `bc_step(u)`                                     |
| `define_excitation`  | `wave.py`        | `excitation_kernel` | `excitation_step(u, signal, t_index)`            |
| `define_get_signal`  | `wave.py`        | `get_signal_kernel` | `get_signal_step(u, um, t_index)`                |
| `define_set_signal`  | `wave.py`        | `set_signal_kernel` | `set_signal_step(u, um, t_index)`                |
| `define_gradient`    | `sensitivity.py` | `gradient_kernel`   | `gradient_step(g_mass, g_stiff, u0, u1, u2, l1)` |

Conventions shared by all of them

| convention | reason |
|---|---|
| prebuilt `args` list, field slots overwritten per call | one allocation instead of $N$, see above |
| scalars pre-cast to `np.int32` / `sim.dtype` | `RawKernel` marshals by the object's own type, so an untyped Python scalar is rejected outright and a mismatched array dtype is read as garbage |
| grid indices flattened on the host by `flatten_indices` | the padded strides are known at setup, so the device never recomputes them |
| axis extents and strides packed by `axis_geometry` | one ordering of the trailing kernel arguments for every `ndim`, mirrored by the factor-free variant in `boundary.py` and `sensitivity.py` |
| the closure returns the field it wrote | lets the loop read as `u0 = fd_step(u0, u1, u0)` |
### define_step_method
Launches `fd_kernel` over the whole padded grid with `grid_block`, one thread per node
### define_excitation
Flat 1D launch of 256 threads, one thread per source, adding the current signal sample into the field
### define_get_signal
The mirror image of the excitation, one thread per sensor, writing row `t_index` of the recording

`simulate` calls it after the buffer swap, so the recorded row is the field that was just computed
### define_set_signal
The inverse of `define_get_signal`, one thread per node, writing the field back from row `t_index` of a record

Only the reverse march of the [sensitivity](sensitivity.md) analysis needs it, to replay a recorded boundary strip, which is why it assigns where the excitation adds
### define_boundary
see [boundary](boundary.md)
### define_gradient
see [sensitivity](sensitivity.md)

## the staggered lattice

A staggered equation puts its unknowns off the nodes, and the geometry that follows is a
function of `component_offsets` and `PAIRS` alone rather than of the physics, so it lives
here and both [elastic](elastic.md) and [maxwell](maxwell.md) call it

| member | signature | description |
|---|---|---|
| voigt order | `PAIRS` | the $\left(k,l\right)$ pairs per dimension, the shear or curl rows being `PAIRS[ndim][ndim:]` |
| weights | `component_weights(sim, c)`, `pair_weights(sim, axes)` | the cell weights $W$ of a point family, halved on the walls of the axes it is not staggered on |
| averages | `point_average(sim, field, c)`, `pair_average(sim, field, axes)` | the arithmetic two-node mean at a component point and the harmonic four-node mean at a pair point |
| average adjoints | `point_average_adjoint(sim, density, c)`, `pair_average_adjoint(sim, density, field, axes)` | their transposes, which is how a point density is chained back onto the nodal design field |

They are free functions taking `sim` rather than a shared base class: what the two
equations hold in common is the lattice, and their material models, `build_materials` and
`define_step` are precisely what they do not share

## the CUDA prelude

`compile_kernels` prepends two source blocks to every `.cu` before handing it to
`RawModule`: the coefficient tables from [stencils](stencils.md) `preamble`, then
`kernels/common.cuh`. The prelude carries what every equation repeats, so a `.cu` holds
only its own operator: the `real_t` typedef, the `OP_W` / `CLOSURE` and `SG_W` accessors
with their `__constant__ tables`, `NPAIRS` / `PAIR_ROW`, `rad_node` / `rad_half`,
`clamped_face`, and the byte-identical `excitation_kernel`, `get_signal_kernel` and
`set_signal_kernel`

Injecting it as source rather than as an include keeps the module cache correct without an
include path, exactly as the stencil table already did. What stays per equation is the
`AXIS_*` and `INTERIOR_OR_RETURN` macro block, which is **not** common: the anisotropic
adjoint takes no per-axis factors where the others do, so one shared definition would
change its kernel signatures rather than deduplicate them
