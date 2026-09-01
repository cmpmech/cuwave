# CUDA anisotropic

The forward kernels of the [anisotropic](anisotropic.md) equation, gathering the cell assembly at each node

## compilation logic

| flag | set from | effect |
|---|---|---|
| `USE_FLOAT` | `precision` | `real_t` is `float` rather than `double` |
| `NDIM` | `len(Nx)` | selects the interior guard, the geometry macro and the loop bounds |
| `USE_DAMPING` | `damping is not None` | adds the damped update and its two extra arguments |

`RADIUS` stays at its default of 1, since the class caps `space_order` at 2. `BLK` is $2\,\textrm{RADIUS}$, `CELLS` is $\textrm{BLK}^\textrm{ndim}$, the cells a node borders and equally the nodes a cell reaches, and `NLOC` is `CELLS * NDIM`, the side of the stencil table. All are compile-time, so every loop over them unrolls fully

Unlike [cuda_scalar](cuda_scalar.md) neither `STENCIL_RADIUS` nor its coefficient tables enter: the table arrives as the `stencil` argument, one cell's contribution to the nodal stencil at $\gamma=1$

## macros

### INTERIOR_OR_RETURN
Declares `a0`, `a1`, `a2` and `idx` and returns on the ghost ring and the padding tail, exactly as in [cuda_scalar](cuda_scalar.md)

### AXIS_GEOM
Declares three compile-time arrays the cell loop indexes rather than unrolling by hand: `A` the per-axis grid index, `S` the per-axis stride over the padded shape (last is 1), `NN` the per-axis logical extent

## device functions

### cell_inside
**aim**
decide whether cell `c` lies inside the domain, and where its low corner sits
**input args**
`c`: cell code, bit $d$ set on the plus side of axis $d$; `A`: grid index per axis; `NN`: logical extent per axis; `S`: stride per axis
**output args**
`base`: offset from `idx` to the cell's low corner; `radius`: 1, the interior grading being inert at this order; `self`: the node's own entry of that cell
**how?**
- the cell exists when both of its corners do, on every axis, which for the node gathering it is
$$\textrm{bit}_d(c)=1:\quad a_d<N_d-2,\qquad\textrm{bit}_d(c)=0:\quad a_d>1$$
- restricting the assembly to the cells that pass is what makes $\boldsymbol{\sigma}\cdot\mathbf{n}=0$ the natural boundary condition, so no ghost value of $\mathbf{u}$ is ever read and the ring the [boundary](boundary.md) kernels would fill is not needed

### block_offset
**aim**
offset of entry `l` of cell `c`, relative to the node gathering that cell
**input args**
`c`: cell code; `l`: entry code, in the same bit convention; `radius`, `S`: as above
**how?**
$$\textrm{offset}=\sum_d\left(\textrm{bit}_d(c)-1+\textrm{bit}_d(l)\right)s_d$$

## kernels

### fd_kernel
**parallelization**
one thread per node of the padded grid, retired to the interior by the guard
**input args**
`u0`, `u1`: the fields at $t-2$ and $t-1$, each `NDIM` components of `cs` entries laid out back to back; `minv`: lumped $1/\left(\gamma\rho_0VW\right)$, nodal and scalar; `cell`: the harmonic cell mean $\gamma_c$, stored at the cell's low corner; `stencil`: `NLOC` by `NLOC`, the cell's contribution at $\gamma=1$, row-major; `damping`: nodal $d$, with `USE_DAMPING` only; `dt`: $\Delta t$, with `USE_DAMPING` only; `cs`: component stride, the padded node count; `f0`: $\Delta t^2$, the only axis factor the kernel reads; `N0, N1, N2`: logical extents, ghost nodes in and padding out; `s0, s1`: strides of axes 0 and 1
**output args**
`u2`: the field at $t$, same layout as `u1`
**internal args**
`force`: the internal force per component, accumulated over the node's cells; `gc`: $\gamma_c$ of the cell being gathered; `self`: the node's own entry of that cell; `mi`: inverse inertia at the node
**how?**
- the force gathers the node's row of the stencil from every cell it borders
$$f_i=-\sum_{c\ni i}\gamma_c\sum_{l,j}\hat{K}_{\left(\textrm{self},i\right)\left(l,j\right)}u_j^{l}$$
- the march is the same three-term recursion the pressure equation uses, applied component by component since the lumped mass is diagonal and shared
$$u_i^{t}=-u_i^{t-2}+2u_i^{t-1}+\Delta t^2\,m^{-1}f_i$$
- with `USE_DAMPING` the update divides through by $1+\beta$ as in [cuda_scalar](cuda_scalar.md), one damping field serving every component
$$\beta=\frac{d\,\Delta t}{2m}$$

Note that `f1` and `f2` are accepted and never read: the tail `wave.axis_geometry` packs is per-axis, while the elastic factors live in the stencil table instead

### excitation_kernel
Byte-identical to [cuda_scalar](cuda_scalar.md). A vector source is `NDIM` entries of `lin_index`, one per component, at `d * cs + idx`, so the `atomicAdd` covers a source and a sensor landing on the same node and component

### get_signal_kernel
Byte-identical to [cuda_scalar](cuda_scalar.md), the component folded into `lin_index` the same way

### set_signal_kernel
Byte-identical to [cuda_scalar](cuda_scalar.md), assignment rather than `atomicAdd`, so it restores a recorded state
