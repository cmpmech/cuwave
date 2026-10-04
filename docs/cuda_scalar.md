# CUDA scalar

## compilation logic
`compile_kernels(sim, path)` builds one `cp.RawModule` per setup: the kernels are written once for any dimension, precision and order, and the specifics arrive as `nvcc` flags or as injected source

| flag                          | set from                                                              | effect                                                        |
| ----------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------- |
| `-DNDIM=1\|2\|3`              | `sim.ndim`                                                            | selects the number of stencil axes                            |
| `-DUSE_FLOAT`                 | `precision == "float32"`                                              | `real_t = float`, otherwise `double`                          |
| `-DUSE_DAMPING`               | `sim.compile_flags`, once `Simulation.damping` is set | compiles the damping term and its two extra kernel parameters |
| `-DUSE_DOMAIN`                | `sim.compile_flags`, once `Simulation.domain` is set  | launches `fd_kernel` over a tile list and grades the cells at the domain wall, two extra kernel parameters |
| `-DUSE_STREAM`, `-DSTREAM_TX` | `sim.compile_flags`, where `PressureWave.streams` (a wide unmasked 2D stencil on a full grid) | compiles the streaming `fd_kernel` below instead of the per-thread one, its tile `STREAM_TX = threads[-1]` wide |
| `STENCIL_RADIUS`, `OP_COEFFS` | `stencils.preamble(space_order)`                                      | the radius $R$ and the cell-coefficient table                 |
| `--use_fast_math`             | always                                                                |                                                               |

Two source blocks are **prepended to the text** rather than passed as `-D` flags: the stencil table, then `kernels/common.cuh`. `RawModule` caches on the code string, so a different `space_order` is a different module automatically, without the order having to appear in the options, and the prelude needs no include path

The prelude is where the `real_t` typedef, the `OP_W` and `CLOSURE` accessors and the three transfer kernels below actually live, so every `.cu` in the repo carries only its own operator. [wave](wave.md) lists what it holds and what it deliberately does not

`stencils.preamble(4)` for instance emits
```c
#define STENCIL_RADIUS 2
#define OP_COEFFS { {1.0, 0.0}, {1.25, -0.08333333333333333} }
```
which are the two cell-flux stencils, written out. Note that they approximate the derivative at the **cell** midpoint $x_{i+1/2}$, not on the node $x_i$
$$\textrm{row 1 (radius 1, order 2):}\qquad D_i^+=u_{i+1}-u_i$$
$$\textrm{row 2 (radius 2, order 4):}\qquad D_i^+=\frac{5}{4}\left(u_{i+1}-u_i\right)-\frac{1}{12}\left(u_{i+2}-u_{i-1}\right)$$
with $D_i^+/\Delta x\approx\partial u/\partial x$ at $x_{i+1/2}$, and $D_i^-=D_{i-1}^+$ the flux through the opposite cell
## finite difference helpers

### OP_W, OP_C
contain the finite difference weights:
- `OP_C(r,k)` with `r` as stencil order and `k` as stencil index: $r, k\in[0,R-1]$
- `OP_W(r,k)` makes it 1-index, so that $r,k\in[1,R]$
- $R$ is the (maximum) `STENCIL_RADIUS`
### CLOSURE
computes the current stencil radius based on the distance `a` to the boundary in the axis direction
- `a` and `N-1-a` are the distances to the two ghost nodes, so the radius grades **down** towards a wall: the first interior node gets radius 1 (order 2), the deep interior saturates at $R$
### flux_divergence_axis
**aim**
compute $\nabla\cdot(k\nabla u)$ at $i$  (`idx`)
**input args**
`u1`: array of $u$; `stiff`: array of $k$; `idx`: index; `s`: stride along axis; `uc`: `u1[idx]`; `sc`: `sc[idx]`; `factor`: condensed prefactor $2\Delta t^2/h^2$ from `step_factors` (see below), with $h$ as node distance in axis; `r`: radius of finite difference scheme in axis
**internal args**
`sp, sm`: `stiff` at the two neighbours $k_{i\pm1}$; `gp, gm`: cell stiffnesses $k_{i\pm\frac{1}{2}}$ as the *halved* harmonic mean of `sc` with `sp`/`sm`; `Dp, Dm`: the two cell fluxes $D_i^+, D_i^-$, initialized with the $k=1$ term that every radius shares and completed in the loop
**how?**
- approximation of outer gradient (flux divergence)
$$\nabla\cdot(k\nabla u)|_i\approx \frac{1}{h} (k_{i+\frac{1}{2}}\nabla u_{i+\frac{1}{2}}-k_{i-\frac{1}{2}}\nabla u_{i-\frac{1}{2}})$$
- $k_{i+\frac{1}{2}}, k_{i-\frac{1}{2}}$ approximated via harmonic mean (`stiff`): the series average, which keeps the flux single-valued across a material jump; putting $k$ on the cell is what lets the scheme avoid ever differentiating it
- approximation of inner gradients with higher order finite differences
$$\nabla_{i+\frac{1}{2}}\approx \frac{1}{h}\sum_{k=1}^r w_{r,k}(u_{i+k}-u_{i-(k-1)})$$
$$\nabla_{i-\frac{1}{2}}\approx \frac{1}{h}\sum_{k=1}^r w_{r,k}(u_{i+(k-1)}-u_{i-k})$$
 - `factor` captures the condensed factor $\frac{1}{h^2}$ of the two nested differences, the $\Delta t^2$ of the time step (and $c_0^2$ for `ScalarWave`), plus the **2** that `gp, gm` are missing (they are $k_ik_{i\pm1}/(k_i+k_{i\pm1})$, so the effective cell stiffness is the full harmonic mean $2k_ik_{i\pm1}/(k_i+k_{i\pm1})$)
## domain helpers
Compiled only under `-DUSE_DOMAIN`; the derivation of the cell radii is in [scalar](scalar.md)
### flux_divergence_cells
**aim**
`flux_divergence_axis` with each of the two cells at its own radius, zero for a cell leaving the domain
**input args**
as `flux_divergence_axis`, with `rp, rm`: the radii of the cells $i\pm\frac{1}{2}$ in place of the one `r`, or `DIRICHLET` (7) for a cell open onto a held node
**how?**
- a cell of radius 0 contributes nothing, so neither its stiffness nor the node across it is read: the zero-flux wall
- a `DIRICHLET` cell contributes $\mp k_iu_i$ (with `factor`'s 2 that is $k_i(0-u_i)/(h/2)$ per $h$), a zero at the cell midpoint; again nothing across it is read
- otherwise the same inner gradient and harmonic mean, the full one written out rather than halved into `factor`'s 2
### wall_laplacian
**aim**
the flux divergence summed over the axes for a node near the wall
**input args**
`u1, stiff, idx, uc, sc`: as `flux_divergence_axis`; `code`: the node's entry of `cells`, bits $6d..6d+2$ the radius `rp` and $6d+3..6d+5$ the radius `rm` of axis $d$; `f0, f1, f2`, `s0, s1`: as `fd_kernel`
**how?**
- `__noinline__`, so the registers the rare wall node needs are not reserved for every deep node of the kernel
### BLOCK_X, BLOCK_Y, BLOCK_Z
the block coordinates: `blockIdx` without `-DUSE_DOMAIN`, and with it the 10-bit fields of the block's entry `tile` in the tile list, fastest axis lowest
## boundary helpers

### BC_PARAMS, BC_GEOM
macro pair carrying the dimension-dependent tail of the boundary kernel signature
* `BC_PARAMS` expands to the grid-size (`N0,N1,...`) and stride (`s0,s1,...`) parameters for the current `NDIM`
- `BC_GEOM` expands to local arrays `n[NDIM]` (grid sizes) and `s[NDIM]` (strides, fastest axis fixed to 1), packing the `BC_PARAMS` scalars for dimension-generic indexing: enables `bc_ghost`to keep one loop over axes instead of an `#if` per dimension
### bc_ghost
**aim**
map a flat thread id `t` to the ghost node it owns and to the direction pointing into the domain
**input args**
`t`: flat thread index; `faces`: bitmask with bit $2d+\textrm{side}$ set when face (axis $d$, side) belongs to this launch, side 0 low and 1 high; `n`, `s`: from `BC_GEOM`
**output args**
`ghost`: flat index of the ghost node (by reference); `normal`: signed stride pointing inward, $+s_d$ on the low side and $-s_d$ on the high side (by reference); the return value is `false` for threads past the last node of the launch
## kernels
### fd_kernel
**parallelization**
threads act over entire grid (1D, 2D or 3D)
**aim**
compute next step $u^{n+1}$ based on central difference approximation in space & time
**input args**
`u0`: array of $u^{n-1}$; `u1`: array of $u^n$; `u2`: array of $u^{n+1}$ to be overwritten; `stiff`: array of $k$; `minv`: array of $1/m$; `derive_inertia`: true when `stiff=1/minv`, i.e., $k=m$; `damping`: array of $d$; `dt` time step size; `cells`: the packed cell radii of `domain_cells`, 0 on a node not stepped (outside `domain`, or `dirichlet`) and `DEEP` ($2^{30}$) where every radius is the full one; `tiles`: one int per launched block, its coordinates in 10-bit fields and bit 30 set when the tile holds a node that is not `DEEP`; `f0, f1, f2`: condensed per-axis factors $2\Delta t^2/h_d^2$ (times $c_0^2$ for `ScalarWave`) from `step_factors`; `N0, N1, N2`: logical grid dimensions `sim.Nx`, ghost nodes included and padding excluded; `s0, s1` strides of axis 0 and 1 over the *padded* shape
**internal args**
`a0, a1, a2`: grid indices; `idx`: flat index into the padded array, $a_0s_0+a_1s_1+a_2$; `r0, r1, r2`: finite difference radii in axis directions; `uc, sc`: central grid entries; `laplacian`: spatially approximated laplacian; `mi`: `minv`derived via `stiff`; `beta`: damping term
**how?**
- central difference in time, explicit, with $\Delta t^2$ already folded into $f_d$
$$u^{n+1}\approx\frac{1}{1+\beta}\left(2 u^{n} - u^{n-1}(1-\beta) + \frac{1}{m}\nabla\cdot (k\nabla u^n)\right),\qquad\beta=\frac{d\,\Delta t}{2m}$$
- without `-DUSE_DAMPING` the $\beta$ branch is not compiled at all and the update is
$$u^{n+1}=-u^{n-1}+2u^n+m^{-1}\nabla\cdot(k\nabla u^n)$$
- under `-DUSE_DOMAIN` the grid is a 1D launch over `tiles`, and only a tile with bit 30 reads `cells`: a branch uniform per block, so a tile wholly inside runs the plain stencil and pays nothing. There a node outside returns before writing, which leaves it at the zero it started at, and a node near the wall replaces the plain laplacian with `wall_laplacian`
### fd_kernel (streaming)
Same aim, operator and update as `fd_kernel`, compiled in its place under `-DUSE_STREAM`: the taps are taken in the same order, so the two agree bit for bit at order 2 and to the last bit of a contracted fma above it
**parallelization**
a 2D launch, `blockIdx.x` a tile of `STREAM_TX` columns of axis 1 and `blockIdx.y` a chunk of `CHUNK_ROWS` rows of axis 0, which the block marches row by row
**input args**
as `fd_kernel` without `cells, tiles`, plus `chunk`: the rows of axis 0 a block marches, `CHUNK_ROWS` from `scalar.py`
**internal args**
`q`: register queue of $u^n$ on rows $a_0-R\ldots a_0+R$ of the thread's column; `sm, sc, sp`: $k$ on rows $a_0-1, a_0, a_0+1$; `su, sk`: the current row of $u^n$ and $k$ in shared memory, with a halo of $R$; `old`: $u^{n-1}$, read before the barriers so its latency hides behind the tile loads
**how?**
- each row: shift the queues by one, load row $a_0+R$ of $u$ and row $a_0+1$ of $k$ into them, write the row's tile and its halo, barrier
- axis 0 from the registers, axis 1 by `flux_divergence_axis` on the tile, then the update of `fd_kernel`
- every value is read from memory once per step and the stencil's reuse along axis 0 stays in registers, which is what the plain kernel leaves to L1. That pays for a wide stencil only: below order 6, on a grid too small to fill the device with chunks, and in 3D, L1 serves the per-thread kernel as well, and a mask keeps the per-thread kernel for its tile list, which the stream would not fit

### homogeneous_neumann_kernel
**parallelization**
flat 1D launch, 256 threads per block, one thread per ghost node (the ghost ring is not a grid-shaped domain, so these kernels decode a linear thread id via `bc_ghost` instead of using the grid mapping of `fd_kernel`)
**aim**
enforce zero flux, $\partial u/\partial n=0$, on every face of the launch
**input args**
`u`: array of $u$, ghost ring overwritten in place; `faces`: bitmask of the faces this launch owns; `BC_PARAMS`: grid extents and strides
**how?**
- `u[ghost] = u[ghost + 2 * normal]`, i.e. $0\leftarrow2$ and $N-1\leftarrow N-3$: mirroring about the wall node makes the central difference across the wall vanish
### homogeneous_dirichlet_kernel
Same parallelization, input args and grouping as `homogeneous_neumann_kernel`, with the odd mirror in place of the even one
- `u[ghost] = -u[ghost + 2 * normal]`: antisymmetry about the wall node, the pressure-release or free-surface condition, and reflection comes back with flipped sign
- homogeneous in the exact sense: the wall node is never written by this kernel, but the odd mirror makes its own `fd_kernel` update self-annihilating ($D^+-D^-=-2u_1$ at radius 1), so from $u^0=u^1=0$ it stays at exactly $0$ for all $n$

The three transfer kernels below live in `kernels/common.cuh` and not in `scalar.cu`, since every equation shares them byte for byte; a fourth there, `adjoint_excitation_kernel`, is documented with its caller in [cuda_scalar_sensitivity](cuda_scalar_sensitivity.md). They are documented here because the scalar case is the one with no component index to fold in, and the other `cuda_*` pages list only what differs

### excitation_kernel
**parallelization**
flat 1D launch, 256 threads per block, one thread per source
**aim**
add the scaled signal sample of time step `t_index` into the field
**input args**
`u`: array of $u$, accumulated into; `source`: the whole $(N,\,\textrm{num sources})$ record; `offset`: `t_index * num_sources`, the row start; `lin_index`: flat indices of the source nodes; `num_sources`; `weight`: per-source $\Delta t^2/m$ times `source_factor`
**how?**
- `weight` is precomputed once at setup by `excitation_weights`, so this kernel needs neither a material field nor any knowledge of the formulation
- `lin_index` comes from `wave.flatten_indices`, which collapses the $(\textrm{ndim},\textrm{num})$ grid indices into flat indices of the padded array once, outside the time loop
- `source` is passed whole with an `offset` rather than as a row view: slicing a row per step costs more launch overhead on the host than this kernel takes on the device
- `atomicAdd` rather than `+=`, because nothing stops two entries of `lin_index` from naming the same node (a plain read-modify-write would then silently lose one of them)

### get_signal_kernel
**parallelization**
flat 1D launch, 256 threads per block, one thread per sensor
**aim**
write row `t_index` of the $(N,\,\textrm{num sensors})$ record `um` from the current field
**input args**
`u`: array of $u$; `um`: the record, written to; `offset`: `t_index * num_sensors`, the row start; `lin_index`: flat indices of the sensor nodes; `num_sensors`
**how?**
- a plain gather, `um[offset + idx] = u[lin_index[idx]]`, the mirror image of `excitation_kernel` (no atomic needed, since two sensors on the same node write the same value to different slots)
- called after the buffer swap, so it records the field just computed, $u^{n+1}$

### set_signal_kernel
Same parallelization and input args as `get_signal_kernel` with the two arrays swapping roles, `u[lin_index[idx]] = um[offset + idx]`, so the record is read and the field written
**how?**
- assignment rather than the `atomicAdd` of `excitation_kernel`, because it restores a recorded state instead of adding a contribution to one: the node ends at the stored value whatever it held before
- the primitive the reverse march of `reconstruction_sensitivity` replays its boundary strip through, see [sensitivity](sensitivity.md)
