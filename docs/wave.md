# Wave
a wave simulation is set up as follows:
1. define wave equation (& parametrization) with `Simulation`
	- `PressureWave`
		- `ScalarWave`
		- `AcousticWave`
	- `ElasticWave`
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

derived to specific wave physics
- `PressureWave`
- `ElasticWave`

where `PressureWave` has specific coefficient parametrizations for
- `ScalarWave`
- `AcousticWave`

A specialization adds the physics: it turns one design field `indicator` $\gamma$ into the nodal coefficient fields the kernels read, and supplies the constants that fold $\Delta t$, $\Delta x$ and the material scaling into the update

| member             | signature                                  | description                                                                                                                                                    |
| ------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| materials          | `build_materials(indicator)` | evaluates the parametrization on $\gamma$ and returns the nodal fields the kernels read, `damping` among them, mirroring their ghost ring into the value the reflecting wall implies |
| coefficients       | `parametrization(indicator)`               | maps $\gamma$ onto the coefficient fields, returning `None` for the one the kernel derives                                                                     |
| derivative         | `parametrization_jacobian()`               | the pair $\left(\partial m/\partial\gamma,\,\partial k/\partial\gamma\right)$ the [sensitivity](sensitivity.md) analysis contracts the adjoint with            |
| per-axis factors   | `step_factors()`                           | the constant multiplying the flux difference along each axis                                                                                                   |
| source scaling     | `source_factor()`                          | the constant the excitation carries on top of $\Delta t^2/m$                                                                                                   |
| kernel arguments   | `step_kernel_args(mat)`                    | packs the material fields into the argument tuple of the step kernel                                                                                           |
| excitation weights | `excitation_weights(mat, lin_index)`       | evaluates $\Delta t^2/m$ at the source nodes only, so a derived inertia field is never formed over the whole grid, carrying the damped update's divisor $1/\left(1+\beta\right)$ with $\beta=d\,\Delta t/2m$ so the injected impulse does not depend on $d$                                              |
| kernel options     | `compile_flags`                            | adds `-DUSE_DAMPING` once `damping` is set                                                                                          |
### PressureWave
$$m\ddot{u}+d\dot{u}-\nabla\cdot\left(k\nabla u\right)=f$$
with the inertia $m$, the damping $d$ and the stiffness $k$, stored as the nodal fields `minv` $=1/m$, `damping` and `stiff` $=k$, since the kernel only ever needs the inverse inertia

`damping` is a constructor field and not a per-call argument, so every path that takes a `Simulation` — `simulate`, `sensitivity` and the [utils](utils.md) glue over them — steps the same operator and no two call sites can disagree about it. `None` is the lossless default, and is what `superposition_sensitivity` requires

`derive_inertia` marks the parametrizations in which $m=k$, so that the fields carry the impedance only and the wave speed sits in `step_factors`: `minv` is then recovered from `stiff` in the kernel, saving one field pass per step and one grid field of memory

#### ScalarWave
$$\gamma\ddot{u}+d\dot{u}-c_0^2\nabla\cdot\left(\gamma\nabla u\right)=\frac{f}{\rho_0}$$
with the background wave speed `wavespeed` $c_0$ and density `density` $\rho_0$, where $\gamma$ scales the density $\rho=\gamma\rho_0$ and hence the impedance, leaving the wave speed at $c_0$ everywhere

| member | specific to `ScalarWave` |
|---|---|
| `parametrization(indicator)` | $k=\gamma$ and $m$ derived, as $\gamma$ scales inertia and stiffness alike (`derive_inertia`) |
| `parametrization_jacobian()` | $\left(1,\,1\right)$ |
| `step_factors()` | $2\,c_0^2\Delta t^2/\Delta x_k^2$, carrying the wave speed |
| `source_factor()` | $1/\rho_0$ |

#### AcousticWave
$$\frac{1}{\kappa}\ddot{u}+d\dot{u}-\nabla\cdot\left(\frac{1}{\rho}\nabla u\right)=f$$
$$\frac{1}{\rho\left(\gamma\right)}=\frac{1}{\rho_1}+\gamma\left(\frac{1}{\rho_2}-\frac{1}{\rho_1}\right),\qquad\frac{1}{\kappa\left(\gamma\right)}=\frac{1}{\kappa_1}+\gamma\left(\frac{1}{\kappa_2}-\frac{1}{\kappa_1}\right)$$
with the two materials `rho1`/`kappa1` and `rho2`/`kappa2` that $\gamma$ interpolates between, $\gamma=0$ giving the first and $\gamma=1$ the second, and the wave speed $c=\sqrt{\kappa/\rho}$ following from the two fields

| member | specific to `AcousticWave` |
|---|---|
| `parametrization(indicator)` | $m=1/\kappa$ and $k=1/\rho$, both stored as the inverse the kernel wants, so neither $\rho$ nor $\kappa$ is ever formed |
| `parametrization_jacobian()` | $\left(1/\kappa_2-1/\kappa_1,\,1/\rho_2-1/\rho_1\right)$, constant since both coefficients are affine in $\gamma$ |
| `step_factors()` | $2\,\Delta t^2/\Delta x_k^2$, the wave speed already sitting in the fields |
| `source_factor()` | $1$ |

### ElasticWave
$$m\ddot{\mathbf{u}}+d\dot{\mathbf{u}}-\nabla\cdot\boldsymbol{\sigma}=\mathbf{f},\qquad\boldsymbol{\sigma}=\lambda\,\textrm{tr}\left(\boldsymbol{\varepsilon}\right)\mathbf{I}+2\mu\boldsymbol{\varepsilon},\qquad\boldsymbol{\varepsilon}=\frac{1}{2}\left(\nabla\mathbf{u}+\nabla\mathbf{u}^\top\right)$$
with the Lamé parameters $\lambda$ and $\mu$, solving for the displacement vector $\mathbf{u}$ instead of a scalar

**TODO** not implemented yet

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
