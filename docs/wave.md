# Wave
a wave simulation is set up as follows:
1. define wave equation (& parametrization) with `Simulation`
	- `PressureWave`
		- `ScalarWave`
		- `AcousticWave`
	- `ElasticWave`
2. define source with `Source`
3. define `indicator` (if heterogeneous)
4. define postprocessing
5. run simulation with `simulate`
## simulate
`simulate` has two jobs:
1. preparation of the kernels
	- `fd_step` (spatial discretization)
	- `bc_step` (modification for boundary conditions)
	- `excitation_step` (source contribution)
2. unrolling of the time integration (where kernels are performed at each time step)

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
| materials          | `build_materials(indicator, damping=None)` | evaluates the parametrization on $\gamma$ and returns the nodal fields the kernels read, mirroring their ghost ring into the value the reflecting wall implies |
| coefficients       | `parametrization(indicator)`               | maps $\gamma$ onto the coefficient fields, returning `None` for the one the kernel derives                                                                     |
| derivative         | `parametrization_jacobian()`               | the pair $\left(\partial m/\partial\gamma,\,\partial k/\partial\gamma\right)$ the [sensitivity](sensitivity.md) analysis contracts the adjoint with            |
| per-axis factors   | `step_factors()`                           | the constant multiplying the flux difference along each axis                                                                                                   |
| source scaling     | `source_factor()`                          | the constant the excitation carries on top of $\Delta t^2/m$                                                                                                   |
| kernel arguments   | `step_kernel_args(mat)`                    | packs the material fields into the argument tuple of the step kernel                                                                                           |
| excitation weights | `excitation_weights(mat, lin_index)`       | evaluates $\Delta t^2/m$ at the source nodes only, so a derived inertia field is never formed over the whole grid                                              |
| kernel options     | `compile_flags`                            | adds `-DUSE_DAMPING` once `build_materials` was given a damping field                                                                                          |
### PressureWave
$$m\ddot{u}+d\dot{u}-\nabla\cdot\left(k\nabla u\right)=f$$
with the inertia $m$, the damping $d$ and the stiffness $k$, stored as the nodal fields `minv` $=1/m$, `damping` and `stiff` $=k$, since the kernel only ever needs the inverse inertia

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

TODO not implemented yet

## define_\[kernel]
