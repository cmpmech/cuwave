// Prepended by cpu.RawModule in place of the CUDA toolkit (docs/cpu.md).
// Compile-time configuration:
//   CUWAVE_SCAN

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

using std::max;
using std::min;

struct dim3 {
  unsigned x = 1, y = 1, z = 1;
};
static thread_local dim3 threadIdx, blockIdx;
static dim3 blockDim, gridDim;

#ifdef CUWAVE_SCAN
#define __global__ CUWAVE_KERNEL
#define __constant__ CUWAVE_GLOBAL
#else
#define __global__
#define __constant__
#endif
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#define __restrict__ __restrict
#define __launch_bounds__(...)
#define __maxnreg__(...)

// ----------------------------------- intrinsics
#define __fmul_rn(a, b) ((float)(a) * (float)(b))
#define __dmul_rn(a, b) ((double)(a) * (double)(b))
#define __fmaf_rn(a, b, c) std::fma((float)(a), (float)(b), (float)(c))
#define __fma_rn(a, b, c) std::fma((double)(a), (double)(b), (double)(c))

inline float __int_as_float(const int i) {
  float f;
  std::memcpy(&f, &i, sizeof f);
  return f;
}

// a warp of one thread: every vote is the thread's own
#define __activemask() 1u
#define __all_sync(lanes, p) (p)

template <class T> inline T atomicAdd(T *p, const T v) {
  T old;
#pragma omp atomic capture
  {
    old = *p;
    *p += v;
  }
  return old;
}

// no block-wide memory or barrier: those kernels are the GPU's alone
#define __shared__ CUWAVE_CPU_HAS_NO_SHARED_MEMORY
#define __syncthreads CUWAVE_CPU_HAS_NO_BLOCK_BARRIER
#define __syncthreads_or CUWAVE_CPU_HAS_NO_BLOCK_BARRIER

// ------------------------------------- launches
template <class... A, std::size_t... I>
inline void cuwave_call(void (*kernel)(A...), void **args,
                        std::index_sequence<I...>) {
  kernel(*static_cast<std::remove_cv_t<A> *>(args[I])...);
}

// the blocks of `grid` across the OpenMP threads, each block's threads in turn
template <class... A>
inline void cuwave_launch(void (*kernel)(A...), const dim3 grid,
                          const dim3 block, void **args) {
  gridDim = grid;
  blockDim = block;
  const long blocks = (long)grid.x * grid.y * grid.z;
#pragma omp parallel for schedule(static)
  for (long b = 0; b < blocks; ++b) {
    blockIdx = dim3{unsigned(b % grid.x), unsigned(b / grid.x % grid.y),
                    unsigned(b / ((long)grid.x * grid.y))};
    for (unsigned z = 0; z < block.z; ++z)
      for (unsigned y = 0; y < block.y; ++y)
        for (unsigned x = 0; x < block.x; ++x) {
          threadIdx = dim3{x, y, z};
          cuwave_call(kernel, args, std::index_sequence_for<A...>{});
        }
  }
}

#define CUWAVE_LAUNCH(name)                                                    \
  extern "C" void launch_##name(unsigned gx, unsigned gy, unsigned gz,          \
                                unsigned bx, unsigned by, unsigned bz,          \
                                void **args) {                                  \
    cuwave_launch(name, dim3{gx, gy, gz}, dim3{bx, by, bz}, args);              \
  }

#define CUWAVE_SIZEOF(name)                                                    \
  extern "C" const unsigned long sizeof_##name = sizeof(name);
