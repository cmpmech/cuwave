# Scalar

**PressureWave** solves the scalar pressure wave equation for one unknown per node, the cheapest of the three [wave](wave.md) families and the one every driver reaches for first, with `ScalarWave` and `AcousticWave` supplying the two parametrizations of its coefficient fields

$$m\ddot{u}+d\dot{u}-\nabla\cdot\left(k\nabla u\right)=f$$

with the inertia $m$, the damping $d$ and the stiffness $k$, stored as the nodal fields `minv` $=1/m$, `damping` and `stiff` $=k$, since the kernel only ever needs the inverse inertia

One unknown per node and a per-axis flux, so the operator needs no `component_offsets` and no cell gather: the vector siblings [elastic](elastic.md) and [anisotropic](anisotropic.md) pay for the cross terms this equation does not have

`damping` is a constructor field and not a per-call argument, so every path that takes a `Simulation` (`simulate`, `sensitivity` and the [utils](utils.md) glue over them) steps the same operator and no two call sites can disagree about it. `None` is the lossless default, and is what `superposition_sensitivity` requires

`derive_inertia` marks the parametrizations in which $m=k$, so that the fields carry the impedance only and the wave speed sits in `step_factors`: `minv` is then recovered from `stiff` in the kernel, saving one field pass per step and one grid field of memory

The hooks a parametrization fills in are the ones [wave](wave.md) lists for every specialization; the two below differ only in what they put into them

## ScalarWave
$$\gamma\ddot{u}+d\dot{u}-c_0^2\nabla\cdot\left(\gamma\nabla u\right)=\frac{f}{\rho_0}$$
with the background wave speed `wavespeed` $c_0$ and density `density` $\rho_0$, where $\gamma$ scales the density $\rho=\gamma\rho_0$ and hence the impedance, leaving the wave speed at $c_0$ everywhere

| member | specific to `ScalarWave` |
|---|---|
| `parametrization(indicator)` | $k=\gamma$ and $m$ derived, as $\gamma$ scales inertia and stiffness alike (`derive_inertia`) |
| `parametrization_jacobian()` | $\left(1,\,1\right)$ |
| `step_factors()` | $2\,c_0^2\Delta t^2/\Delta x_k^2$, carrying the wave speed |
| `source_factor()` | $1/\rho_0$ |

## AcousticWave
$$\frac{1}{\kappa}\ddot{u}+d\dot{u}-\nabla\cdot\left(\frac{1}{\rho}\nabla u\right)=f$$
$$\frac{1}{\rho\left(\gamma\right)}=\frac{1}{\rho_1}+\gamma\left(\frac{1}{\rho_2}-\frac{1}{\rho_1}\right),\qquad\frac{1}{\kappa\left(\gamma\right)}=\frac{1}{\kappa_1}+\gamma\left(\frac{1}{\kappa_2}-\frac{1}{\kappa_1}\right)$$
with the two materials `rho1`/`kappa1` and `rho2`/`kappa2` that $\gamma$ interpolates between, $\gamma=0$ giving the first and $\gamma=1$ the second, and the wave speed $c=\sqrt{\kappa/\rho}$ following from the two fields

| member | specific to `AcousticWave` |
|---|---|
| `parametrization(indicator)` | $m=1/\kappa$ and $k=1/\rho$, both stored as the inverse the kernel wants, so neither $\rho$ nor $\kappa$ is ever formed |
| `parametrization_jacobian()` | $\left(1/\kappa_2-1/\kappa_1,\,1/\rho_2-1/\rho_1\right)$, constant since both coefficients are affine in $\gamma$ |
| `step_factors()` | $2\,\Delta t^2/\Delta x_k^2$, the wave speed already sitting in the fields |
| `source_factor()` | $1$ |

