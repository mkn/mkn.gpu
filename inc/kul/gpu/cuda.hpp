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
#ifndef _KUL_GPU_CUDA_HPP_
#define _KUL_GPU_CUDA_HPP_

#include <cuda_runtime.h>

#include "kul/log.hpp"
#include "kul/assert.hpp"
#include "kul/gpu/tuple.hpp"

#include "kul/gpu/cuda/def.hpp"

#define KUL_GPU_ASSERT(x) (KASSERT((x) == cudaSuccess))

namespace kul::gpu {

template <typename T>
static constexpr bool is_floating_point_v = std::is_floating_point_v<T> or
                                           std::is_same_v<_Float16, T>;


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

template <typename T, typename SIZE = uint32_t>
struct DeviceMem {
  using Pointers = kul::Pointers<T, SIZE>;

  DeviceMem(){}
  DeviceMem(SIZE _s) : s(_s) {
    SIZE alloc_bytes = s * sizeof(T);
    KLOG(OTH) << "GPU alloced: " << alloc_bytes;
    if(s)
      KUL_GPU_ASSERT(cudaMalloc((void**)&p, alloc_bytes));
  }

  DeviceMem(T const* const t, SIZE _s) : DeviceMem(_s) {send(t, _s);}
  DeviceMem(std::vector<T> const& v) : DeviceMem(&v[0], v.size()) {}
  DeviceMem(std::vector<T> && v) : DeviceMem(v) {}
  DeviceMem(Pointers const& p) : DeviceMem(p.p, p.s) {}

  ~DeviceMem() { if (p && s) KUL_GPU_ASSERT(cudaFree(p)); }

  void send(Pointers const& p, SIZE start = 0){
    send(p.p, p.s, start);
  }

  void send(T const * const t, SIZE _size = 1, SIZE start = 0){
    KUL_GPU_ASSERT(cudaMemcpy(p + start, t, _size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void send(std::vector<T> && v, SIZE start = 0){
    send(&v[0], v.size(), start);
  }

  void fill_n(T t, SIZE _size, SIZE start = 0){
    // TODO - improve with memSet style
    send(std::vector<T>(_size, t), start);
  }

  decltype(auto) operator+(size_t size){
    DeviceMem<T> view; // has no size;
    view.p = this->p + size;
    return view;
  }

  template <typename Container>
  Container& take(Container& c) const {
    KUL_GPU_ASSERT(cudaMemcpy(&c[0], p, s * sizeof(T), cudaMemcpyDeviceToHost));
    return c;
  }
  template <typename Container = std::vector<T>>
  Container take() const {
    Container c(s);
    return take(c);
  }
  decltype(auto) operator()() const { return take(); }
  auto& size() const { return s; }
  SIZE s = 0;
  T* p = nullptr;
};

template <bool GPU>
struct ADeviceClass {};

template <>
struct ADeviceClass<true> {};

template <>
struct ADeviceClass<false> {
  ~ADeviceClass() { invalidate(); }

  void _alloc(void* ptrs, uint8_t size) {
    KUL_GPU_ASSERT(cudaMalloc((void**)&ptr, size));
    KUL_GPU_ASSERT(cudaMemcpy(ptr, ptrs, size, cudaMemcpyHostToDevice));
  }

  template <typename as, typename... DevMems>
  decltype(auto) alloc(DevMems&&... mem) {
    if (ptr) throw std::runtime_error("already malloc-ed");
    auto ptrs = make_pointer_container(mem.p...);
    if (sizeof(as) != sizeof(ptrs))
       throw std::runtime_error("VERY NO");

    _alloc(&ptrs, sizeof(ptrs));
    return static_cast<as*>(ptr);
  }

  void invalidate() {
    if (ptr) {
      KUL_GPU_ASSERT(cudaFree(ptr));
      ptr = nullptr;
    }
  }

  void* ptr = nullptr;
};

template <bool GPU>
struct DeviceClass : ADeviceClass<GPU> {
  template <typename T, typename SIZE = uint32_t>
  using container_t = std::conditional_t<GPU, T*, kul::gpu::DeviceMem<T, SIZE>>;
};

void sync(){
  KUL_GPU_ASSERT(cudaDeviceSynchronize());
}

struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(size_t w, size_t h, size_t tpx, size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}
  Launcher(size_t x, size_t y, size_t z, size_t tpx, size_t tpy, size_t tpz)
      : Launcher{dim3(x / tpx, y / tpy, z / tpz), dim3(tpx, tpy, tpz)} {}

  template <typename F, typename... Args>
  void operator()(F f, Args... args) {
    kul::gpu::sync();

    f<<<g, b, ds, s>>>(args...);
  }
  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  cudaStream_t s = 0;
};


template<typename T, typename V>
void fill_n(kul::gpu::DeviceMem<T> & p, size_t size, V val){
  p.fill_n(val, size);
}

template<typename T, typename V>
void fill_n(kul::gpu::DeviceMem<T> && p, size_t size, V val){
  fill_n(p, size, val);
}


} /* namespace kul::gpu */


#undef KUL_GPU_ASSERT
#endif /* _KUL_GPU_CUDA_HPP_ */
