/**
Copyright (c) 2020, Philip Deegan.
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
// IWYU pragma: private, include "kul/gpu.hpp"
#ifndef _KUL_GPU_ROCM_HPP_
#define _KUL_GPU_ROCM_HPP_

#include "hip/hip_runtime.h"

#include "kul/log.hpp"
#include "kul/span.hpp"
#include "kul/tuple.hpp"
#include "kul/assert.hpp"

#include "kul/gpu/def.hpp"

#define KUL_GPU_ASSERT(x) (KASSERT((x) == hipSuccess))

namespace kul::gpu {
#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
namespace hip {
#define KUL_GPU_NS kul::gpu::hip
#else
#define KUL_GPU_NS kul::gpu
#endif  // KUL_GPU_FN_PER_NS

struct Stream {
  Stream() { KUL_GPU_ASSERT(result = hipStreamCreate(&stream)); }
  ~Stream() { KUL_GPU_ASSERT(result = hipStreamDestroy(stream)); }

  auto& operator()() { return stream; };

  void sync() { result = hipStreamSynchronize(stream); }

  hipError_t result;
  hipStream_t stream;
};

template <typename Size>
void alloc(void*& p, Size size) {
  KUL_GPU_ASSERT(hipMalloc((void**)&p, size));
}

template <typename T, typename Size>
void alloc(T*& p, Size size) {
  KLOG(TRC) << "GPU alloced: " << size * sizeof(T);
  KUL_GPU_ASSERT(hipMalloc((void**)&p, size * sizeof(T)));
}

template <typename T, typename Size>
void alloc_host(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  KUL_GPU_ASSERT(hipHostMalloc((void**)&p, size * sizeof(T)));
}

void destroy(void* p) { KUL_GPU_ASSERT(hipFree(p)); }

template <typename T>
void destroy(T* ptr) {
  KUL_GPU_ASSERT(hipFree(ptr));
}

template <typename T>
void destroy_host(T* ptr) {
  KUL_GPU_ASSERT(hipHostFree(ptr));
}

template <typename Size>
void send(void* p, void* t, Size size = 1) {
  KUL_GPU_ASSERT(hipMemcpy(p, t, size, hipMemcpyHostToDevice));
}

template <typename T, typename Size>
void send(T* p, T const* t, Size size = 1, Size start = 0) {
  KUL_GPU_ASSERT(hipMemcpy(p + start, t, size * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T, typename Size>
void take(T* p, T* t, Size size = 1, Size start = 0) {
  KUL_GPU_ASSERT(hipMemcpy(p + start, t, size * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T, typename Size>
void send_async(T* p, T const* t, Stream& stream, Size size = 1, Size start = 0) {
  KUL_GPU_ASSERT(hipMemcpyAsync(p + start,              //
                                t + start,              //
                                size * sizeof(T),       //
                                hipMemcpyHostToDevice,  //
                                stream()));
}

template <typename T, typename Span>
void take_async(T* p, Span& span, Stream& stream, std::size_t start) {
  static_assert(kul::is_span_like_v<Span>);
  KUL_GPU_ASSERT(hipMemcpyAsync(span.data(),              //
                                p + start,                //
                                span.size() * sizeof(T),  //
                                hipMemcpyDeviceToHost,    //
                                stream()));
}

void sync() { KUL_GPU_ASSERT(hipDeviceSynchronize()); }

#include "kul/gpu/device.hpp"

template <typename F, typename... Args>
void launch(F f, dim3 g, dim3 b, std::size_t ds, hipStream_t& s, Args&&... args) {
  KLOG(TRC);
  std::apply(
      [&](auto&&... params) { hipLaunchKernelGGL(f, g, b, ds, s, params...); },
      devmem_replace(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()));
}

// https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/HIP-GUIDE.html#calling-global-functions
struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(size_t w, size_t h, size_t tpx, size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}
  Launcher(size_t x, size_t y, size_t z, size_t tpx, size_t tpy, size_t tpz)
      : Launcher{dim3(x / tpx, y / tpy, z / tpz), dim3(tpx, tpy, tpz)} {}

  template <typename F, typename... Args>
  void operator()(F f, Args&&... args) {
    launch(f, g, b, ds, s, args...);
  }

  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  hipStream_t s = 0;
};

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

#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
} /* namespace hip */
#endif  // KUL_GPU_FN_PER_NS
} /* namespace kul::gpu */

#undef KUL_GPU_ASSERT
#endif /* _KUL_GPU_ROCM_HPP_ */
