// https://raw.githubusercontent.com/NVIDIA-developer-blog/code-samples/master/series/hip-cpp/overlap-data-transfers/async.cu

/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hip/hip_runtime.h"

#include <stdio.h>

// Convenience function for checking HIP runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline void checkHip([[maybe_unused]] hipError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
#endif
  // return result;
}

__global__ void kernel(float* a, int offset) {
  int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  float x = (float)i;
  float s = sinf(x);
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s * s + c * c);
}

float maxError(float* a, int n) {
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i] - 1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

int main(int argc, char** argv) {
  int const blockSize = 256, nStreams = 4;
  int const n = 4 * 1024 * blockSize * nStreams;
  int const streamSize = n / nStreams;
  int const streamBytes = streamSize * sizeof(float);
  int const bytes = n * sizeof(float);

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  hipDeviceProp_t prop;
  checkHip(hipGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkHip(hipSetDevice(devId));

  // allocate pinned host memory and device memory
  float *a, *d_a;
  checkHip(hipHostMalloc((void**)&a, bytes));  // host pinned
  checkHip(hipMalloc((void**)&d_a, bytes));    // device

  float ms;  // elapsed time in milliseconds

  // create events and streams
  hipEvent_t startEvent, stopEvent, dummyEvent;
  hipStream_t stream[nStreams];
  checkHip(hipEventCreate(&startEvent));
  checkHip(hipEventCreate(&stopEvent));
  checkHip(hipEventCreate(&dummyEvent));
  for (int i = 0; i < nStreams; ++i) checkHip(hipStreamCreate(&stream[i]));

  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkHip(hipEventRecord(startEvent, 0));
  checkHip(hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice));
  kernel<<<n / blockSize, blockSize>>>(d_a, 0);
  checkHip(hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost));
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkHip(
        hipMemcpyAsync(&d_a[offset], &a[offset], streamBytes, hipMemcpyHostToDevice, stream[i]));
    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    checkHip(
        hipMemcpyAsync(&a[offset], &d_a[offset], streamBytes, hipMemcpyDeviceToHost, stream[i]));
  }
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 2:
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkHip(
        hipMemcpyAsync(&d_a[offset], &a[offset], streamBytes, hipMemcpyHostToDevice, stream[i]));
  }
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkHip(
        hipMemcpyAsync(&a[offset], &d_a[offset], streamBytes, hipMemcpyDeviceToHost, stream[i]));
  }
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // cleanup
  checkHip(hipEventDestroy(startEvent));
  checkHip(hipEventDestroy(stopEvent));
  checkHip(hipEventDestroy(dummyEvent));
  for (int i = 0; i < nStreams; ++i) checkHip(hipStreamDestroy(stream[i]));
  checkHip(hipFree(d_a));
  checkHip(hipHostFree(a));

  return 0;
}
