/**
Copyright (c) 2019, Philip Deegan.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Philip Deegan nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _KUL_GPU_ROCM_HPP_
#define _KUL_GPU_ROCM_HPP_

#include "kul/log.hpp"
#include "hip/hip_runtime.h"
#define GPU_ASSERT(x) (assert((x) == hipSuccess))

namespace kul::gpu {

template <typename T>
__global__ void vectoradd(T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c,
                          size_t width, size_t height) {
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  int z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  int i = (z * height) + (y * width) + x;
  a[i] = b[i] + c[i];
}

void prinfo(size_t dev = 0) {
  // https://rocm-developer-tools.github.io/HIP/group__Device.html
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, dev);

  KOUT(NON) << " System version " << devProp.major << "." << devProp.minor;
  KOUT(NON) << " agent name     " << devProp.name;
  KOUT(NON) << " cores          " << devProp.multiProcessorCount;
  KOUT(NON) << " threadsPCore   " << devProp.maxThreadsPerMultiProcessor;
  KOUT(NON) << " TotalMem       " << (devProp.totalGlobalMem / 1000000) << " MB";
  KOUT(NON) << " BlockMem       " << (devProp.sharedMemPerBlock / 1000) << " KB";
  KOUT(NON) << " warpSize       " << devProp.warpSize;
  KOUT(NON) << " threadsPBlock  " << devProp.maxThreadsPerBlock;
}

template <typename T>
struct DeviceMem {
  DeviceMem(size_t _s) : s(_s) { GPU_ASSERT(hipMalloc((void**)&p, s * sizeof(T))); }
  DeviceMem(std::vector<T> const& v) : DeviceMem(v.size()) {
    GPU_ASSERT(hipMemcpy(p, &v[0], v.size() * sizeof(T), hipMemcpyHostToDevice));
  }
  ~DeviceMem() {
    if (p) GPU_ASSERT(hipFree(p));
  }
  template <typename C>  // container
  C& xfer(C& c) {
    GPU_ASSERT(hipMemcpy(&c[0], p, s * sizeof(T), hipMemcpyDeviceToHost));
    return c;
  }
  template <typename C = std::vector<T>>  // container
  C get() {
    C c(s);
    return xfer(c);
  }
  size_t s = 0;
  T* p = nullptr;
};

// https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/HIP-GUIDE.html#calling-global-functions
struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(size_t w, size_t h, size_t tpx, size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}
  template <typename F, typename... Args>
  void operator()(F f, Args... args) {
    hipLaunchKernelGGL(f, g, b, ds, s, args...);
  }
  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  hipStream_t s = 0;
};

} /* namespace kul::gpu */
#endif /* _KUL_LOG_HPP_ */
