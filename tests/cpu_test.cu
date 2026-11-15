// Compiled by tests/cpu_test.py through cpu.RawModule, for the host only.
// Compile-time configuration:
//   OFFSET
//   SHARED

#ifndef OFFSET
#define OFFSET 0
#endif

__constant__ double TABLE[4];

// -------------------------------------- kernels
extern "C" {
__global__ void index_kernel(int *__restrict__ out, const int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const int nx = gridDim.x * blockDim.x, ny = gridDim.y * blockDim.y;
  const int i = (z * ny + y) * nx + x; // the thread's place in the whole launch
  if (i < n)
    out[i] = i + OFFSET;
}

// -----------------------------------------------------------------------------------
__global__ void scalar_kernel(double *__restrict__ out, const int a,
                              const float b, const double c,
                              const long long d) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = d;
  }
}

// -----------------------------------------------------------------------------------
__global__ void table_kernel(double *__restrict__ out) {
  out[threadIdx.x] = TABLE[threadIdx.x];
}

// -----------------------------------------------------------------------------------
__global__ void count_kernel(int *__restrict__ total) { atomicAdd(total, 1); }

#ifdef SHARED
// -----------------------------------------------------------------------------------
__global__ void shared_kernel(float *__restrict__ out) {
  __shared__ float tile[32];
  tile[threadIdx.x] = threadIdx.x;
  __syncthreads();
  out[threadIdx.x] = tile[31 - threadIdx.x];
}
#endif
}
