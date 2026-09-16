# CUDA maxwell

The forward kernels of the vector curl-curl [maxwell](maxwell.md) equation: one pass forming the curl on its staggered pair points, one pass gathering the transposed curl into the update. The 2D reductions run on [cuda_scalar](cuda_scalar.md) instead and reach none of this

## compilation logic

| flag | set from | effect |
|---|---|---|
| `NDIM` | `len(Nx)` | selects the interior guard and the geometry macros; 1 is an `#error` |
| `USE_DAMPING` | `damping is not None` | adds the damped update and its two extra arguments |
| `USE_MAGNETIC` | `magnetic` | the pair coefficient arrives as a field rather than a scalar |

`real_t`, `STENCIL_RADIUS`, `SG_W`, `NPAIRS`, `PAIR_ROW`, `rad_half` and the three transfer kernels all come from the prelude; see [cuda_elastic](cuda_elastic.md), which documents them

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

The `h` scratch is allocated once by `define_step` and reused every step, the same trade [cuda_elastic](cuda_elastic.md) makes: each curl value is read by up to $2r$ updates, so forming it once costs one round of global traffic instead of recomputing every curl per reader
