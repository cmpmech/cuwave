# CPU

**The CPU fallback** runs cuwave without an NVIDIA GPU: numpy holds the arrays in place of cupy, and the CUDA kernels are compiled unchanged for the host, behind a header that maps the CUDA builtins onto C++ and OpenMP

`backend` picks the device once, at import, from `CUWAVE_DEVICE`, so a driver that sets it does so before its first `cuwave` import (`examples/forward/scalar2D_cpu.py`)

| `CUWAVE_DEVICE` | device |
|---|---|
| unset | the GPU where cupy finds one, else the CPU |
| `cpu` | the CPU, even beside a GPU |
| `gpu` | the GPU, raising where cupy or a device is missing |

Every module imports `cp`, `cpt` and `ndi` from `backend` and calls them as before. On the CPU `cp` is the module `cpu`: numpy for every name it does not define, plus `asnumpy` (a copy, as cupy's is), `get_array_module`, the CUDA runtime the package touches (`cuda.Stream`, `cuda.Event`, `cuda.get_current_stream`, all no-ops on a synchronous host) and `RawModule`. `ndi` is `scipy.ndimage` there, hence the `cpu` extra of the package, and `backend.ON_GPU` tells the two devices apart

## Kernels
`RawModule` has the surface of `cupy.RawModule` the package uses

| member | signature | description |
|---|---|---|
| constructor | `RawModule(code, options=())` | compiles `code` with the `-D` flags of `options`, dropping `--use_fast_math` and raising on any other flag |
| kernel | `get_function(name)` | the kernel `name`, launched as cupy launches one, `kernel(grid, block, args)`: an array by pointer, raising unless it is C-contiguous, a scalar by value, a Python int as int64 and a float as float64 |
| module global | `get_global(name)` | the writable memory of a `__constant__` table, which `AnisotropicElasticWave` fills from the host |

`kernels/cpu.hpp` stands in for the CUDA toolkit. `threadIdx` and `blockIdx` are thread-local, a launch deals its blocks out to the OpenMP threads and runs the threads of each block in turn, `atomicAdd` is an OpenMP atomic, and a warp is a single thread, so a vote (`__all_sync`) returns the thread's own value and `warp_deep` picks each node's wall path alone. The rounding intrinsics `__fmul_rn` and `__fma_rn` become a plain product and `std::fma`

A kernel that shares memory within a block, or synchronizes the block, cannot run one thread at a time, so `__shared__` and `__syncthreads` do not compile. The blocked, fused, streamed and marched steps are therefore the GPU's alone: each `define_blocked_step`, `_fuses_pair`, `streams` and `_refills_in_step` declines on the CPU, and every equation takes its plain step, which those reproduce bitwise. The CUDA graphs of `march` and the side stream of `reconstruction_sensitivity` drop out the same way, and `devices` (of `Simulation` or `misfit_gradient`) raises

## Compilation
The compiler is `CXX` (default `c++`) with `CUWAVE_CXXFLAGS` (default `-O3 -march=native -fopenmp -ffp-contract=off`), to which `-std=c++17 -shared -fPIC` are added, so it has to understand GCC's flags: GCC or clang, with OpenMP. A preprocessor pass with `CUWAVE_SCAN` first names the kernels and module globals the `-D` flags keep, for which `launch_<name>` and `sizeof_<name>` are appended to the source, since a kernel under an `#if` the flags drop must not be named. The library is cached under `$XDG_CACHE_HOME/cuwave` (default `~/.cache/cuwave`) by the hash of compiler, flags and source, so each configuration pays its compile (about a second) once per machine, and a concurrent build of the same one is harmless

The host keeps IEEE arithmetic and, under `-ffp-contract=off`, fuses only the products the kernels pin with `FMA_RN`, so two host kernels written alike agree bitwise, as on the GPU, while a CPU run agrees with a GPU run to round-off only. A numpy call can return another memory layout than its cupy twin (`argwhere` is column-major in numpy), which is why a kernel argument has to be C-contiguous. Each block's threads run one after another, which the compiler does not vectorize, and a CPU has far less memory bandwidth than a GPU, so a step costs about two orders of magnitude more than on a GPU: the fallback is for machines without one, for tests and for small grids. `OMP_NUM_THREADS` sets the threads, all cores by default
