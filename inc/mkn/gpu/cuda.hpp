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
// IWYU pragma: private, include "mkn/gpu.hpp"
#ifndef _MKN_GPU_CUDA_HPP_
#define _MKN_GPU_CUDA_HPP_

#include <cuda_runtime.h>

#include "mkn/kul/log.hpp"
#include "mkn/kul/span.hpp"
#include "mkn/kul/tuple.hpp"
#include "mkn/kul/assert.hpp"

#include "mkn/gpu/def.hpp"

#define MKN_GPU_ASSERT(x) (KASSERT((x) == cudaSuccess))

namespace mkn::gpu {
#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
namespace cuda {
#define MKN_GPU_NS mkn::gpu::cuda
#else
#define MKN_GPU_NS mkn::gpu
#endif  // MKN_GPU_FN_PER_NS

struct Stream {
  Stream() { MKN_GPU_ASSERT(result = cudaStreamCreate(&stream)); }
  ~Stream() { MKN_GPU_ASSERT(result = cudaStreamDestroy(stream)); }

  auto& operator()() { return stream; };

  void sync() { result = cudaStreamSynchronize(stream); }

  cudaError_t result;
  cudaStream_t stream;
};

template <typename Size>
void alloc(void*& p, Size size) {
  MKN_GPU_ASSERT(cudaMalloc((void**)&p, size));
}

template <typename T, typename Size>
void alloc(T*& p, Size size) {
  KLOG(TRC) << "GPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(cudaMalloc((void**)&p, size * sizeof(T)));
}

template <typename T, typename Size>
void alloc_host(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(cudaMallocHost((void**)&p, size * sizeof(T)));
}

template <typename T, typename Size>
void alloc_managed(T*& p, Size size) {
  KLOG(TRC) << "GPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(cudaMallocManaged((void**)&p, size * sizeof(T)));
}

void destroy(void* p) { MKN_GPU_ASSERT(cudaFree(p)); }

template <typename T>
void destroy(T*& ptr) {
  MKN_GPU_ASSERT(cudaFree(ptr));
}

template <typename T>
void destroy_host(T*& ptr) {
  MKN_GPU_ASSERT(cudaFreeHost(ptr));
}

template <typename Size>
void send(void* p, void* t, Size size = 1) {
  MKN_GPU_ASSERT(cudaMemcpy(p, t, size, cudaMemcpyHostToDevice));
}

template <typename T, typename Size>
void send(T* p, T const* t, Size size = 1, Size start = 0) {
  MKN_GPU_ASSERT(cudaMemcpy(p + start, t, size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T, typename Size>
void take(T* p, T* t, Size size = 1, Size start = 0) {
  MKN_GPU_ASSERT(cudaMemcpy(t, p + start, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, typename Size>
void send_async(T* p, T const* t, Stream& stream, Size size = 1, Size start = 0) {
  MKN_GPU_ASSERT(cudaMemcpyAsync(p + start,               //
                                 t + start,               //
                                 size * sizeof(T),        //
                                 cudaMemcpyHostToDevice,  //
                                 stream()));
}

template <typename T, typename Span>
void take_async(T* p, Span& span, Stream& stream, std::size_t start) {
  static_assert(mkn::kul::is_span_like_v<Span>);
  MKN_GPU_ASSERT(cudaMemcpyAsync(span.data(),              //
                                 p + start,                //
                                 span.size() * sizeof(T),  //
                                 cudaMemcpyDeviceToHost,   //
                                 stream()));
}

void sync() { MKN_GPU_ASSERT(cudaDeviceSynchronize()); }

#include "mkn/gpu/device.hpp"

template <typename F, typename... Args>
void launch(F&& f, dim3 g, dim3 b, std::size_t ds, cudaStream_t& s, Args&&... args) {
  KLOG(TRC);
  std::apply(
      [&](auto&&... params) { f<<<g, b, ds, s>>>(params...); },
      devmem_replace(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()));
}

//
struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(size_t w, size_t h, size_t tpx, size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}
  Launcher(size_t x, size_t y, size_t z, size_t tpx, size_t tpy, size_t tpz)
      : Launcher{dim3(x / tpx, y / tpy, z / tpz), dim3(tpx, tpy, tpz)} {}

  template <typename F, typename... Args>
  void operator()(F&& f, Args&&... args) {
    launch(f, g, b, ds, s, args...);
  }

  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  cudaStream_t s = 0;
};

//
void prinfo(size_t dev = 0) {
  cudaDeviceProp devProp;
  [[maybe_unused]] auto ret = cudaGetDeviceProperties(&devProp, dev);
  KOUT(NON) << " System version " << devProp.major << "." << devProp.minor;
  KOUT(NON) << " agent name     " << devProp.name;
  KOUT(NON) << " cores          " << devProp.multiProcessorCount;
  KOUT(NON) << " threadsPCore   " << devProp.maxThreadsPerMultiProcessor;
  KOUT(NON) << " TotalMem       " << (devProp.totalGlobalMem / 1000000) << " MB";
  KOUT(NON) << " BlockMem       " << (devProp.sharedMemPerBlock / 1000) << " KB";
  KOUT(NON) << " warpSize       " << devProp.warpSize;
  KOUT(NON) << " threadsPBlock  " << devProp.maxThreadsPerBlock;
}

template <typename T, std::int32_t alignment = 32>
class ManagedAllocator {
  using This = ManagedAllocator<T, alignment>;

 public:
  using pointer = T*;
  using reference = T&;
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    using other = ManagedAllocator<U, alignment>;
  };

  T* allocate(std::size_t const n) const {
    if (n == 0) return nullptr;

    T* ptr;
    alloc_managed(ptr, n);
    if (!ptr) throw std::bad_alloc();
    return ptr;
  }

  void deallocate(T* const p) noexcept {
    if (p) destroy(p);
  }
  void deallocate(T* const p, std::size_t /*n*/) noexcept {  // needed from std::
    deallocate(p);
  }

  bool operator!=(This const& that) const { return !(*this == that); }

  bool operator==(This const& /*that*/) const {
    return true;  // stateless
  }
};

#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
} /* namespace cuda */
#endif  // MKN_GPU_FN_PER_NS
} /* namespace mkn::gpu */

#undef MKN_GPU_ASSERT
#endif /* _MKN_GPU_CUDA_HPP_ */
