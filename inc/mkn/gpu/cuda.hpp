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

#include <vector>

#include <cuda_runtime.h>

#include "mkn/kul/log.hpp"
#include "mkn/kul/span.hpp"
#include "mkn/kul/tuple.hpp"
#include "mkn/kul/assert.hpp"

#include "mkn/gpu/def.hpp"

// #define MKN_GPU_ASSERT(x) (KASSERT((x) == cudaSuccess))

#define MKN_GPU_ASSERT(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
#define MKN_GPU_NS mkn::gpu::cuda
#else
#define MKN_GPU_NS mkn::gpu
#endif  // MKN_GPU_FN_PER_NS

namespace MKN_GPU_NS {

struct Stream {
  Stream() { MKN_GPU_ASSERT(result = cudaStreamCreate(&stream)); }
  ~Stream() { MKN_GPU_ASSERT(result = cudaStreamDestroy(stream)); }

  auto& operator()() { return stream; };

  void sync() { result = cudaStreamSynchronize(stream); }

  cudaError_t result;
  cudaStream_t stream;
};

template <typename T>
struct Pointer {
  Pointer(T* _t) : t{_t} { MKN_GPU_ASSERT(cudaPointerGetAttributes(&attributes, t)); }

  bool is_unregistered_ptr() const { return attributes.type == 0; }
  bool is_host_ptr() const {
    return attributes.type == 1 || (is_unregistered_ptr() && t != nullptr);
  }
  bool is_device_ptr() const { return is_managed_ptr() || attributes.type == 2; }
  bool is_managed_ptr() const { return attributes.type == 3; }

  T* t;
  cudaPointerAttributes attributes;
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

void destroy(void* p) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaFree(p));
}

template <typename T>
void destroy(T*& ptr) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaFree(ptr));
}

template <typename T>
void destroy_host(T*& ptr) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaFreeHost(ptr));
}

template <typename Size>
void send(void* p, void* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpy(p, t, size, cudaMemcpyHostToDevice));
}

template <typename T, typename Size>
void send(T* p, T const* t, Size size = 1, Size start = 0) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpy(p + start, t, size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T, typename Size>
void take(T const* p, T* t, Size size = 1, Size start = 0) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpy(t, p + start, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, typename Size>
void copy(T* dst, T const* src, Size size = 1, Size start = 0) {
  KLOG(TRC);
  Pointer p{dst};
  if (p.is_host_ptr())
    take(src, dst, size, start);
  else
    send(dst, src, size, start);
}

template <typename Type, typename Alloc0, typename Alloc1>
void copy(std::vector<Type, Alloc0>& dst, std::vector<Type, Alloc1> const& src) {
  KLOG(TRC);
  assert(dst.size() >= src.size());
  copy(dst.data(), src.data(), dst.size());
}

template <typename T, typename Size>
void send_async(T* p, T const* t, Stream& stream, Size size = 1, Size start = 0) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpyAsync(p + start,               //
                                 t + start,               //
                                 size * sizeof(T),        //
                                 cudaMemcpyHostToDevice,  //
                                 stream()));
}

template <typename T, typename Span>
void take_async(T* p, Span& span, Stream& stream, std::size_t start) {
  KLOG(TRC);
  static_assert(mkn::kul::is_span_like_v<Span>);
  MKN_GPU_ASSERT(cudaMemcpyAsync(span.data(),              //
                                 p + start,                //
                                 span.size() * sizeof(T),  //
                                 cudaMemcpyDeviceToHost,   //
                                 stream()));
}

void sync() { MKN_GPU_ASSERT(cudaDeviceSynchronize()); }

#include "mkn/gpu/alloc.hpp"
#include "mkn/gpu/device.hpp"

template <typename F, typename... Args>
void launch(F&& f, dim3 g, dim3 b, std::size_t ds, cudaStream_t& s, Args&&... args) {
  std::size_t N = (g.x * g.y * g.z) * (b.x * b.y * b.z);
  KLOG(TRC) << N;
  std::apply(
      [&](auto&&... params) { f<<<g, b, ds, s>>>(params...); },
      devmem_replace(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()));
  sync();
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
    launch(std::forward<F>(f), g, b, ds, s, args...);
  }

  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  cudaStream_t s = 0;
};

struct GLauncher : public Launcher {
  GLauncher(std::size_t s, size_t dev = 0) : Launcher{dim3{}, dim3{}}, count{s} {
    [[maybe_unused]] auto ret = cudaGetDeviceProperties(&devProp, dev);

    b.x = devProp.maxThreadsPerBlock;
    g.x = s / b.x;
    if ((s % b.x) > 0) ++g.x;
  }

  std::size_t count = 0;
  cudaDeviceProp devProp;
};

template <typename F, typename... Args>
__global__ static void global_gd_kernel(F f, std::size_t s, Args... args) {
  if (auto i = mkn::gpu::cuda::idx(); i < s) f(args...);
}

#include "launchers.hpp"

template <typename T, typename V>
__global__ void _vector_fill(T* a, V t, std::size_t s) {
  if (auto i = mkn::gpu::cuda::idx(); i < s) a[i] = t;
}

template <typename Container, typename T>
void fill(Container& c, size_t size, T val) {
  GLauncher{c.size()}(_vector_fill<typename Container::value_type, T>, c.data(), val, size);
}

template <typename Container, typename T>
void fill(Container& c, T val) {
  GLauncher{c.size()}(_vector_fill<typename Container::value_type, T>, c.data(), val, c.size());
}

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

}  // namespace MKN_GPU_NS

#undef MKN_GPU_ASSERT
#endif /* _MKN_GPU_CUDA_HPP_ */
