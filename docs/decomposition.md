# Decomposition

**Decomposition** splits one grid over several GPUs along axis 0, so a problem can outgrow a single device in memory and run on several at once. It is switched on with one setting, `Simulation.devices`, and nothing else in a driver changes: the `DEVICES` setting of `examples/forward/scalarND.py` and `examples/sensitivity/scalar2D_sensitivity.py` is all it takes

Each device steps one **subdomain**: an ordinary `Simulation` over the rows it owns plus $H$ rows of each neighbour, the **halo**, stepped by the unmodified kernels. The subdomain sees a wall where its rows end, so the rows next to that wall are wrong. The error travels inward by at most `sim.reach` rows a step, and every $K$ steps an exchange overwrites each halo with the rows its owner computed. The owned rows therefore never see it, and they come out bitwise equal to the run on one device:
$$H=2K\,r_\textrm{reach}+p+2\qquad K=6$$
with the halo `halo_rows(sim)` $H$, the exchange interval `EVERY` $K$, the one-step read reach `sim.reach` $r_\textrm{reach}$ and the `space_order` $p$. One interval of reach covers the forward march, and $p+2$ the rows the cut wall's own stencil spoils. The second interval covers a strip a reverse march replays, which was itself recorded with up to one interval of error. $K$ is a multiple of the slot rotation periods 2 and 3, which is what lets `march` still replay its chunks as a captured graph

An exchange refreshes only the outer `stale_rows(sim)` $=K\,r_\textrm{reach}+p+2$ rows of each halo, the ones an interval can have spoilt, since every deeper row is still right. Inside `replaying()`, which `reconstruction_sensitivity` wraps around its reverse march, it refreshes the whole halo, because the replayed strip brings back values that were stale when they were recorded; `superposition_sensitivity` wraps its backward march too, whose first step runs outside it

| member | signature | description |
|---|---|---|
| switch | `Simulation(..., devices=(0, 1))` | the GPUs axis 0 is split over, evenly, `None` for the current device alone |
| split functions | `simulate`, `sensitivity`, `reconstruction_sensitivity`, `superposition_sensitivity` | run each subdomain in a thread of its own, and through them `measure`, `misfit` and `misfit_gradient` |
| refused | `spectral_sensitivity`, `source_sensitivity` | raise `NotImplementedError` under `devices` |
| halo | `halo_rows(sim)` | the rows held past each cut |
| layout | `partition(sim)` | the rows each subdomain holds and owns, raising where a share is thinner than the halo or a halo reaches an axis-0 pml |
| march hook | `exchange(state)` | refreshes the halo of every array a march carries, a no-op outside a split run |
| replay | `replaying()` | context in which an exchange refreshes the whole halo rather than its stale rows |
| collective | `agree(value, reduce=min)` | one value every subdomain shares, so a schedule decided per subdomain stays in step |

Inputs and outputs keep the single-device shapes: the indicator, the source and the sensors are global, and the returned field, gradients and record are global on the calling device. A host `indicator` returns the fields on the host instead, which is how a grid larger than the calling device is assembled at all. Sources and sensors are injected wherever a subdomain holds them, halo included, and the objective sees the whole record once, so any objective works unchanged

What a march carries is passed to it as `state`: the field slots and the pml memories the step closures expose as `memory`, the same set a checkpoint snapshots, and the halo rows of a stored history window. A memory along axis 0 lives in the end subdomains only, so `partition` keeps it clear of every halo instead of exchanging it

## shots or rows

`measure`, `misfit` and `misfit_gradient` also take `devices`, which deals the shots out instead (see [utils](utils.md)). The two compose and answer different questions. Shots need no exchange and scale with the number of shots, so they win whenever a shot fits one device. Rows pay for the halo, $2H$ extra rows per cut and an exchange every $K$ steps, which only a grid too large or too slow for one device repays

The halo grows with the order and with $K$: at order 8 a staggered elastic grid holds about 100 rows past each cut, so a share has to be several times thicker than that before the split pays. The exchange goes through the host where the GPUs have no peer access, so its cost depends on the interconnect, and it is left to the caller to choose the split
