// Compiled on its own by regularization.BrushGenerator; no configuration flags.
// Each kernel combines a node's taps at `offsets`, (taps, 2) pairs, skipping taps
// outside the array.

// -------------------------------------- kernels
extern "C" {
__global__ void dilate_kernel(const unsigned char *__restrict__ in,
                              unsigned char *__restrict__ out,
                              const int *__restrict__ offsets, const int taps,
                              const int N0, const int N1) {
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;
  if (a0 >= N0 || a1 >= N1)
    return;
  unsigned char any = 0; // every bit plane is dilated on its own
  for (int k = 0; k < taps; ++k) {
    const int b0 = a0 + offsets[2 * k], b1 = a1 + offsets[2 * k + 1];
    if (b0 >= 0 && b0 < N0 && b1 >= 0 && b1 < N1)
      any |= in[b0 * N1 + b1];
  }
  out[a0 * N1 + a1] = any;
}

// -----------------------------------------------------------------------------------
__global__ void maximum_kernel(const float *__restrict__ in,
                               float *__restrict__ out,
                               const int *__restrict__ offsets, const int taps,
                               const int N0, const int N1) {
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;
  if (a0 >= N0 || a1 >= N1)
    return;
  float most = -__int_as_float(0x7f800000); // -inf
  for (int k = 0; k < taps; ++k) {
    const int b0 = a0 + offsets[2 * k], b1 = a1 + offsets[2 * k + 1];
    if (b0 >= 0 && b0 < N0 && b1 >= 0 && b1 < N1)
      most = fmaxf(most, in[b0 * N1 + b1]);
  }
  out[a0 * N1 + a1] = most;
}

// -----------------------------------------------------------------------------------
__global__ void sum_kernel(const float *__restrict__ in, float *__restrict__ out,
                           const int *__restrict__ offsets, const int taps,
                           const int N0, const int N1) {
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;
  if (a0 >= N0 || a1 >= N1)
    return;
  float total = 0.0f;
  for (int k = 0; k < taps; ++k) {
    const int b0 = a0 + offsets[2 * k], b1 = a1 + offsets[2 * k + 1];
    if (b0 >= 0 && b0 < N0 && b1 >= 0 && b1 < N1)
      total += in[b0 * N1 + b1];
  }
  out[a0 * N1 + a1] = total;
}
}
