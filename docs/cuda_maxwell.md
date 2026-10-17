# CUDA maxwell

The forward kernels of the vector curl-curl [maxwell](maxwell.md) equation: one pass forming the curl on its staggered pair points, one pass gathering the transposed curl into the update. The 2D reductions run on [cuda_scalar](cuda_scalar.md) instead and reach none of this

## compilation logic

| flag | set from | effect |
|---|---|---|
| `NDIM` | `len(Nx)` | selects the interior guard and the geometry macros; 1 is an `#error` |
| `USE_DAMPING` | `damping is not None` | adds the damped update and its two extra arguments |
| `USE_MAGNETIC` | `magnetic` | the pair coefficient arrives as a field rather than a scalar |

`real_t`, `STENCIL_RADIUS`, `SG_W`, `NPAIRS`, `PAIR_ROW`, `rad_half` and the transfer kernels all come from the prelude; see [cuda_elastic](cuda_elastic.md), which documents them

A pair index `p` is `PAIR_ROW(k, l) - NDIM`, so the curl scratch holds `NPAIRS` fields back to back in the same order the elastic shear stresses use

## device functions

### curl_component
**aim**
the $\left(k,l\right)$ curl component at its own staggered point
**input args**
`u`: an electric field, `NDIM` components of `cs` entries back to back; `idx`, `A`, `S`, `NN`, `F`: geometry, `F` holding $1/\Delta x_d$; `k`, `l`: the pair axes, `k < l`
**how?**
- both taps are half-point differences, since each term differentiates a component along an axis it is not staggered on, so `rad_half` grades both and `rad_node` never appears
$$b_{kl}=\frac{1}{\Delta x_k}\sum_{j\le r_2}c_j\left(u_l^{a+j}-u_l^{a+1-j}\right)-\frac{1}{\Delta x_l}\sum_{j\le r_1}c_j\left(u_k^{a+j}-u_k^{a+1-j}\right)$$
- byte-identical to `shear_strain` in [cuda_elastic](cuda_elastic.md) but for the sign of the first sum, which is the whole difference between a symmetric gradient and a curl

### pair_weight
**aim**
the cell weight $w_p$ of a pair point
**input args**
`k`, `l`: the pair axes; `A`, `NN`: geometry
**how?**
- a pair point spans a full cell in its own two axes and half a cell on a wall of any other, so the weight is $\tfrac{1}{2}$ per remaining axis the point sits on a wall of: identically 1 in 2D, and $\tfrac{1}{2}$ or 1 in 3D
- computed rather than stored, since it carries no design dependence; under `USE_MAGNETIC` it is folded into `nu_pair` on the host instead

## pml helpers
The layout of [cuda_elastic](cuda_elastic.md) with the normal block deleted: `aux` holds the curl memories $\psi_k$ of the signed $\partial_eE_k$ at the pair point $\left(k,e\right)$, then the force memories $\zeta_c$ of the axis-$e$ part of the force on $E_c$. Every pair point sits half a node up `e`, so the curl memories read the half-point profile and the force memories the node one
### curl_row, Curl
**aim**
`curl_row(c, e)`: the curl row of the pair $\left(c,e\right)$, counted from the first pair; `Curl`: that row `j` nodes along `e` from a node, what the force on $c$ reads along `e`
### curl_term
**aim**
the signed $\partial_eE_k$ at the pair point $\left(k,e\right)$, as `curl_component` takes it: negated when $k<e$, zero for $k=e$ and where the pair point does not exist
**input args**
`u`: any accessor of `(k, j)`, a `Field` forwards and a `Memory` in the adjoint; `k`, `e`: component and axis; `A`, `NN`: node indices and extents; `fe`: $1/\Delta x_e$
### divergence_term
**aim**
the axis-$e$ part of the force on $E_c$, as `fd_kernel` takes it, signed $+$ when $e<c$, with the taps of the pair point each reads
**input args**
`h`: any accessor of `(c, j)`, a `Curl` forwards and a `Memory<Curl>` in the adjoint; `c`, `e`, `A`, `NN`, `fe`: as `curl_term`
### add_curl
**aim**
add $\nu$ times the curl terms `t` of axis `e` onto the pair points, with `pair_weight` or, under `USE_MAGNETIC`, `nu_pair`, as `curl_kernel` weights them

## kernels

### curl_kernel
**parallelization**
one thread per grid point, writing every pair point it carries
**input args**
`u1`: the field at $t-1$; `nu_pair`: `NPAIRS` fields of $w_p\nu_p$, the harmonic pair means, with `USE_MAGNETIC` only; `nu`: the scalar $1/\mu$, used when the permeability is uniform; `cs`: component stride; `f0..f2`: $1/\Delta x_d$; `N0..N2`, `s0, s1`: geometry
**output args**
`h`: `NPAIRS` fields of $w_p\nu b_p$, assigned rather than accumulated; a pair point neither axis carries an unknown for is skipped and keeps the zero the scratch was allocated with, which the gather relies on
**how?**
- one multiply on the curl of each pair point
$$h_p=w_p\,\nu\,b_p$$
- the guard `A[k] > NN[k] - 3 || A[l] > NN[l] - 3` is the pair-point existence test, one staggered ghost per axis of the pair

### fd_kernel
**parallelization**
one thread per grid point, updating every component whose unknown it carries
**input args**
`u0`, `u1`: the fields at $t-2$ and $t-1$; `h`: the weighted curl `curl_kernel` just wrote; `minv`: `NDIM` fields of $1/\left(W_c\bar{\varepsilon}_c\right)$, one per component at its own points, the volume deliberately left out since the force below leaves it out too; `damping`: nodal $\sigma$, with `USE_DAMPING` only; `dt`: $\Delta t$, with `USE_DAMPING` only; `cs`: as above; `f0..f2`: $\Delta t^2/\Delta x_d$
**output args**
`u2`: the field at $t$, same layout as `u1`
**internal args**
`sgn`: $+1$ where the partner axis is below `c` and $-1$ above it; `ap, am`: the pair point of the plus and minus tap; `rp, rm`: that point's own graded radius, which decides whether the tap exists
**how?**
- component $c$ enters the pair $\left(k,l\right)$ as $+\partial_kE_l$ when $c=l$ and as $-\partial_lE_k$ when $c=k$, which is the whole of `sgn`, and the transpose is then the elastic shear gather unchanged
$$f_c=\sum_{m\ne c}\pm\frac{1}{\Delta x_m}\sum_jc_j^{\left(r\left(a\right)\right)}\left(h_{cm}^{i+j-1}-h_{cm}^{i-j}\right)$$
- every tap carries the coefficient of the **pair point it reads**, evaluated by `rad_half` at that point rather than at the thread's own, which is exactly what makes the two launches an exact transpose pair
- the march is the three-term recursion of [cuda_scalar](cuda_scalar.md), component by component, `USE_DAMPING` dividing by $1+\beta$ with one conductivity field serving every component

### pml_curl_kernel, pml_force_kernel, pml_adjoint_curl_kernel, pml_adjoint_force_kernel, pml_adjoint_memory_kernel
Flat 1D launches over the slab of each lined axis: `pml_curl_kernel` after `curl_kernel`, `pml_force_kernel` after `fd_kernel`, their adjoints in the same places, and `pml_adjoint_memory_kernel` last in every adjoint step. They carry the terms [cuda_elastic](cuda_elastic.md) folds into its step kernels (`stress_memory`, `force_memory`, `settle_memory`), with `curl_term`, `divergence_term` and `add_curl` in place of the strain, divergence and stress helpers, and `h` in place of `sigma`
**input args**
flat over the slab rather than the grid, each takes the memory `aux`, its `profile`, the lined axis `e` and the slab rows `lo, hi` that `force_memory` of [cuda_elastic](cuda_elastic.md) takes, with `h`, `nu_pair` and `nu` of `curl_kernel` in place of `sigma` and the elastic material, and no `clamped` mask: a `Conductor` wall is already a zeroed `minv`
**how?**
- forwards, $\psi_k\leftarrow b\,\psi_k+a\,\partial_eE_k$ onto every pair point $\left(k,e\right)$ and $\zeta_c\leftarrow b\,\zeta_c+a\,\partial_eh_{ce}$ into the update; the component along `e` carries neither
- the adjoint pushes each memory back through the other pass, exactly as the elastic one does

The `h` scratch is allocated once by `define_step` and reused every step, the same trade [cuda_elastic](cuda_elastic.md) makes: each curl value is read by up to $2r$ updates, so forming it once costs one round of global traffic instead of recomputing every curl per reader
