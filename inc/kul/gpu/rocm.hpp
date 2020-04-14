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
#include "kul/gpu/rocm/def.hpp"
#define KUL_GPU_ASSERT(x) (assert((x) == hipSuccess))

namespace kul::gpu {

template <typename T>
__global__ void vectoradd(T*  a, const T*  b, const T*  c, size_t width, size_t height) {
  int i = hip::blockDim_i(width, height);
  a[i] = b[i] + c[i];
}

// https://rocm-developer-tools.github.io/HIP/group__Device.html
void prinfo(size_t dev = 0) {
  hipDeviceProp_t devProp;
  [[maybe_unused]] auto ret = hipGetDeviceProperties(&devProp, dev);
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
  DeviceMem(size_t _s) : s(_s) { KUL_GPU_ASSERT(hipMalloc((void**)&p, s * sizeof(T))); }
  DeviceMem(std::vector<T> const& v) : DeviceMem(v.size()) {
    KUL_GPU_ASSERT(hipMemcpy(p, &v[0], v.size() * sizeof(T), hipMemcpyHostToDevice));
  }
  ~DeviceMem() {
    if (p) KUL_GPU_ASSERT(hipFree(p));
  }
  template <typename Container>
  Container& xfer(Container& c) {
    KUL_GPU_ASSERT(hipMemcpy(&c[0], p, s * sizeof(T), hipMemcpyDeviceToHost));
    return c;
  }
  template <typename Container = std::vector<T>>
  Container get() {
    Container c(s);
    return xfer(c);
  }
  auto &size() const { return s; }
  size_t s = 0;
  T* p = nullptr;
};

template <bool GPU, typename gpu_t>
struct ADeviceClass {};

template <typename gpu_t>
struct ADeviceClass<true, gpu_t> {};

template <typename gpu_t>
struct ADeviceClass<false, gpu_t> {

  ~ADeviceClass(){invalidate();}

  auto alloc(gpu_t const& ref){
      if(ptr) throw std::runtime_error("already malloc-ed");
      size = sizeof(ref);
      KUL_GPU_ASSERT(hipMalloc((void**)&ptr, size));
      KUL_GPU_ASSERT(hipMemcpy(ptr, &ref, size, hipMemcpyHostToDevice));
      return ptr;
  }

  void invalidate(){
      if(!ptr) throw std::runtime_error("never malloc-ed");
      KUL_GPU_ASSERT(hipFree(ptr));
  }

  gpu_t* ptr = nullptr;
  size_t size = 0;
};

template <bool GPU, typename gpu_t>
struct DeviceClass : ADeviceClass<GPU, gpu_t>{};

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
#undef KUL_GPU_ASSERT
#endif /* _KUL_LOG_HPP_ */
