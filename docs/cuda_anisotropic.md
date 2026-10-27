# CUDA anisotropic

The forward kernels of the [anisotropic](anisotropic.md) equation, gathering the cell assembly at each node

## compilation logic

| flag | set from | effect |
|---|---|---|
| `USE_FLOAT` | `precision` | `real_t` is `float` rather than `double` |
| `NDIM` | `len(Nx)` | selects the interior guard, the geometry macro and the loop bounds |
| `USE_DAMPING` | `damping is not None` | adds the damped update and its two extra arguments |
| `STENCIL_KEY` | a digest of `dx` and the stiffness, by `AnisotropicElasticWave.compile_flags` | unread: it keys CuPy's module cache, so two simulations whose source and options are otherwise alike never share one `STENCIL` |
| `STEPS`, `TILE0`, `TILE1`, `ROWS` | `BLOCKED` or `LARGE_BLOCKED`, by `AnisotropicElasticWave.define_blocked_step` in 2D float32 without a pml | compiles `blocked_kernel`: the steps a launch advances, the tile and the thread rows sharing it |

`RADIUS` stays at its default of 1, since the class caps `space_order` at 2. `BLK` is $2\,\textrm{RADIUS}$, `CELLS` is $\textrm{BLK}^\textrm{ndim}$, the cells a node borders and equally the nodes a cell reaches, and `NLOC` is `CELLS * NDIM`, the side of the stencil table. All are compile-time, so every loop over them unrolls fully

Unlike [cuda_scalar](cuda_scalar.md) neither `STENCIL_RADIUS` nor its coefficient tables enter: the table, one cell's contribution to the nodal stencil at $\gamma=1$, arrives in the constant array `STENCIL` for `fd_kernel` and `blocked_kernel`, copied into each module once by `AnisotropicElasticWave.define_step` and `define_blocked_step`, and as the `stencil` argument for the sensitivity kernels. Its indices are compile-time in every unrolled loop, so in constant memory each entry is an operand of the fused multiply-add rather than a load, which is what bounds the 3D gather

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

### cell_force
the force `fd_kernel` gathers on every component of a node, below, read from `u` and the cell field strided `cs` a component and `S` an axis, so the plain step reads the global fields and `blocked_kernel` its tile through the same code

### block_offset
**aim**
offset of entry `l` of cell `c`, relative to the node gathering that cell
**input args**
`c`: cell code; `l`: entry code, in the same bit convention; `radius`, `S`: as above
**how?**
$$\textrm{offset}=\sum_d\left(\textrm{bit}_d(c)-1+\textrm{bit}_d(l)\right)s_d$$

## pml helpers
The cell assembly has no per-axis derivative to stretch, so its layer takes the centre point of each cell: a strain memory $\psi_j$ per cell of $\partial_eu_j$ at the cell centre, and a force memory $\zeta_i$ per node of the axis-$e$ part of $\nabla\cdot\sigma_i$, gathered from the same centre stresses. The cell of a slab thread is the one on its wall side, so the low-corner index is $a$ on the low side and $a-1$ on the high one. All lined axes share one `aux` buffer and one `profile` buffer, located per axis by `layers`, because the force memory of one axis reads the stretched stress, which carries every axis' strain memory
### PML_ARGS
the uniform argument list of the five pml kernels, so one launch list serves them all
**input args**
`u1`: the field the pass reads, $u^{t-1}$ forwards and $\lambda^{t+1}$ in the adjoint; `u2`: the field it adds to, $u^t$ or $\lambda^t$; `aux`: the memories of every lined axis back to back, per axis `NDIM` rows of cell memories then `NDIM` of node memories, each `size` long; `minv`, `cell`: as `fd_kernel`; `cv`: the Voigt `C` at $\gamma=1$ times the cell volume, `NVOIGT` by `NVOIGT`; `f`: $\Delta t^2$; `cs`: component stride; `profile`: the `(4, NN[d])` rows of every lined axis back to back; `layers`: per axis `lo, hi`, the start of its memory in `aux` and of its profile in `profile`, `lo = hi = 0` for an unlined axis; `e`: the axis this launch runs on; `f0..f2`: $1/\Delta x_d$; `N0..N2`, `s0, s1`: geometry
### Layer, layer_of, slab_at
**aim**
`Layer`: one axis' memory, profile, slab rows, plane and `size`, unpacked from `layers` by `layer_of`; `slab_at(X, cell, d, l, n, s)`: the slab thread of the node at `X`, or of the cell with low corner `X` (a high-side cell belonging to the node on its wall side), -1 off the slab
### corner_weight, flat
**aim**
`corner_weight(m, d, F)`: the centre-point derivative weight along `d` at corner `m` of a cell, $\pm1/\left(2^{\textrm{ndim}-1}\Delta x_d\right)$ by bit $d$ of `m`; `flat(X, S)`: the flat index of `X`
### cell_gradient, cell_stress, voigt_row
**aim**
`cell_gradient`: $G_{kj}=\partial_ku_j$ at the centre of the cell with low corner `X`, from its $2^\textrm{ndim}$ corners; `cell_stress`: $\gamma\,V\,C\,\textrm{sym}\left(G\right)$ in Voigt rows, the shear strains engineering; `voigt_row(i, k)`: the Voigt row of $\sigma_{ik}$
### own_cell, around
**aim**
`own_cell`: the cell of this slab thread, false where the plane position has none; `around(s, A, NN, X, m)`: cell `s` of the $2^\textrm{ndim}$ around node `A`, its low corner `X` and the corner `m` the node is, false outside the domain
### strain_memory, force_memory
**aim**
row $d$ of the stretched cell gradient: `strain_memory` reads the forward $\psi$ of axis $d$ at a cell, zero off its slab; `force_memory` forms the adjoint's, $\sum_m w_m\,a_m\,\eta_{\zeta,m}$ over the cell's corners, the derivative of the node memory through the node profile
### memory_force
**aim**
add the force at corner `m` of the stress $\gamma VC\,\textrm{sym}\left(P_ep\right)$, the memory `p` placed in row `e` of an otherwise zero gradient
### LAYERS, PML_OR_RETURN
`LAYERS` unpacks one `Layer` per axis into `L[]`; `PML_OR_RETURN` binds the geometry, the layers and this thread's slab node of axis `e` through `SLAB_OR_RETURN(0)` ([cuda_scalar](cuda_scalar.md)), returns past the slab, and names the axis' profile `pe` and memory `me`

## kernels

### fd_kernel
**parallelization**
one thread per node of the padded grid, retired to the interior by the guard
**input args**
`u0`, `u1`: the fields at $t-2$ and $t-1$, each `NDIM` components of `cs` entries laid out back to back; `minv`: lumped $1/\left(\gamma\rho_0VW\right)$, nodal and scalar; `cell`: the harmonic cell mean $\gamma_c$, stored at the cell's low corner; `damping`: nodal $d$, with `USE_DAMPING` only; `dt`: $\Delta t$, with `USE_DAMPING` only; `cs`: component stride, the padded node count; `f0`: $\Delta t^2$, the only axis factor the kernel reads; `N0, N1, N2`: logical extents, ghost nodes in and padding out; `s0, s1`: strides of axes 0 and 1
**output args**
`u2`: the field at $t$, same layout as `u1`
**internal args**
`STENCIL`: `NLOC` by `NLOC` per radius, the cell's contribution at $\gamma=1$, row-major, in constant memory; `force`: the internal force per component, accumulated over the node's cells; `gc`: $\gamma_c$ of the cell being gathered; `self`: the node's own entry of that cell; `mi`: inverse inertia at the node
**how?**
- the force gathers the node's row of the stencil from every cell it borders
$$f_i=-\sum_{c\ni i}\gamma_c\sum_{l,j}\hat{K}_{\left(\textrm{self},i\right)\left(l,j\right)}u_j^{l}$$
- the march is the same three-term recursion the pressure equation uses, applied component by component since the lumped mass is diagonal and shared
$$u_i^{t}=-u_i^{t-2}+2u_i^{t-1}+\Delta t^2\,m^{-1}f_i$$
- with `USE_DAMPING` the update divides through by $1+\beta$ as in [cuda_scalar](cuda_scalar.md), one damping field serving every component
$$\beta=\frac{d\,\Delta t}{2m}$$

Note that `f1` and `f2` are accepted and never read: the tail `wave.axis_geometry` packs is per-axis, while the elastic factors live in the stencil table instead

### blocked_kernel
`STEPS` steps of `fd_kernel` between one load and one store of a tile, the excitation and the record included, bitwise the launches one at a time; 2D float32 only
**parallelization**
a 2D launch of `(TILE1, ROWS)` threads per block, one block per tile of `TILE0` $\times$ `TILE1` nodes; a thread owns `NODES` $=$ `TILE0 / ROWS` consecutive nodes of its column. Tiles overlap by a halo of `HALO` $=$ `STEPS` nodes each side, a cell gather reaching one node a step, and write back only their core
**input args**
`u0, u1`: both components at $t-1, t$; `v0, v1`: the arrays receiving them at $t+S-1, t+S$; `minv, cell` and with `USE_DAMPING` `damping, dt`: as `fd_kernel`; `signal, weights, num_sources, record, num_sensors, t0`, `source_starts, source_entries, sensor_starts, sensor_entries`: as the elastic `blocked_kernel`, entries `component node | point << 16`; `cs`: component stride; `f0, N0, f1, N1, s0`: as `fd_kernel`
**internal args**
`shared`: both components of $u$ twice, swapped every step, and the cell field, each in a guard ring one node wide, read by `cell_force` through the tile strides `T`; `up, uc, mi, beta, tag`: per owned node in registers; `inner`: the owned interior nodes, which the step moves
**how?**
- each step gathers the force off the tile's current $u$ and writes the next into the other copy, so one barrier separates the steps; a node off the interior keeps its value of two steps back, as the plain buffers do
- the edge nodes of a tile read the guard ring and go wrong, one node deeper every step; after `STEPS` steps the error stops short of the core
- the update is `fd_kernel`'s expression, the damped one rounded by `FMA_RN` and `MUL_RN` as nvcc contracts it ($\Delta t\,d/2m$ fused into $1\mp\beta$), and the source term rounded apart as `excitation_kernel` adds it

### pml_cell_kernel
**parallelization**
flat 1D launch over the slab of axis `e`, `PML_THREADS` per block, one cell per thread; every lined axis' launch runs before any `pml_force_kernel`
**aim**
advance the strain memory of axis `e` at the thread's cell
**input args**
see `PML_ARGS`
**how?**
$$\psi_j\leftarrow b_{a+\frac12}\psi_j+a_{a+\frac12}\,G_{ej}\left(u^{t-1}\right)$$
### pml_force_kernel
**parallelization**
as `pml_cell_kernel`, one node per thread, after every `pml_cell_kernel`
**aim**
add the stress of the axis-`e` memory and the advanced force memory to $u^t$
**input args**
see `PML_ARGS`
**internal args**
`force`: the force of $\psi_e$ over the node's cells; `div`: the axis-$e$ part of the divergence of the stretched stress
**how?**
- the stretched gradient adds every lined axis' memory, $\tilde G=G+\sum_dP_d\psi_d$, and the centre stress of it feeds the divergence along `e`
$$\zeta_i\leftarrow b_a\zeta_i-a_a\sum_cw_{m,e}\,\tilde\sigma_{ie},\qquad u_i^t\mathrel{+}=\Delta t^2m^{-1}\left(\textrm{force}_i+\zeta_i\right)$$
### pml_adjoint_memory_kernel
**parallelization**
as `pml_force_kernel`, first in every adjoint step
**aim**
advance the adjoint force memory, $\eta_{\zeta,i}\leftarrow b_a\eta_{\zeta,i}+\lambda_i^{t+1}$
### pml_adjoint_cell_kernel
**parallelization**
as `pml_cell_kernel`, after every `pml_adjoint_memory_kernel`
**aim**
advance the adjoint strain memory: the transpose of `pml_force_kernel`'s read of $u$ and $\psi$
**how?**
- the forward's stretched gradient with $\lambda$ for $u$ and every axis' `force_memory` for $\psi$
$$\eta_{\psi,j}\leftarrow b_{a+\frac12}\eta_{\psi,j}-\gamma V\left[C\,\textrm{sym}\left(G\left(\lambda^{t+1}\right)+\sum_dP_d\,D_d\left(a\,\eta_{\zeta,d}\right)\right)\right]_{ej}$$
### pml_adjoint_force_kernel
**parallelization**
as `pml_force_kernel`, last in every adjoint step
**aim**
add the transposed memory terms of axis `e` to $\lambda^t$
**how?**
$$\lambda_i^t\mathrel{+}=\Delta t^2m^{-1}\sum_c\left(w_{m,e}\,a_{c}\,\eta_{\psi,i}+\textrm{the force of }D_e\left(a\,\eta_\zeta\right)\right)$$
- the forward pair read the cell memory through $D^\top$ and the node memory through the stress, so the adjoint pair swaps them; the update is otherwise the forward one

The layer is centre-point where the interior is fully integrated, so it is a consistent stretching of the interior operator rather than its exact stretched twin; the hourglass mode stays controlled, since a centre gradient neither reads nor drives the checkerboard. `AnisotropicElasticWave` refuses a `C` with a backward wave along a lined axis, see [boundary](boundary.md)

### excitation_kernel
From the shared prelude, so this file does not carry it; see [cuda_scalar](cuda_scalar.md). A vector source is `NDIM` entries of `lin_index`, one per component, at `d * cs + idx`, so the `atomicAdd` covers a source and a sensor landing on the same node and component

### get_signal_kernel
From the shared prelude, the component folded into `lin_index` the same way

### set_signal_kernel
From the shared prelude, assignment rather than `atomicAdd`, so it restores a recorded state
