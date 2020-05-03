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

#include "kul/assert.hpp"

#include "kul/log.hpp"

#include "kul/tuple.hpp"

#include "hip/hip_runtime.h"
#include "kul/gpu/rocm/def.hpp"
#define KUL_GPU_ASSERT(x) (KASSERT((x) == hipSuccess))

namespace kul::gpu {

template <typename T>
static constexpr bool is_floating_point_v = std::is_floating_point_v<T> or
                                           std::is_same_v<_Float16, T>;

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
  DeviceMem(){}
  DeviceMem(size_t _s) : s(_s) {
    size_t alloc_bytes = s * sizeof(T);
    KLOG(OTH) << "GPU alloced: " << alloc_bytes;
    if(s)
      KUL_GPU_ASSERT(hipMalloc((void**)&p, alloc_bytes));
  }

  DeviceMem(T const* const t, size_t _s) : DeviceMem(_s) {send(t, _s);}
  DeviceMem(std::vector<T> && v) : DeviceMem(&v[0], v.size()) {}
  DeviceMem(std::vector<T> const& v) : DeviceMem(&v[0], v.size()) {}
  DeviceMem(kul::Pointers<T> const& p) : DeviceMem(p.p, p.s) {}

  ~DeviceMem() { if (p && s) KUL_GPU_ASSERT(hipFree(p)); }

  void send(T const * const t, size_t _size = 1, size_t start = 0){
    KUL_GPU_ASSERT(hipMemcpy(p + start, t, _size * sizeof(T), hipMemcpyHostToDevice));
  }

  void send(std::vector<T> && v, size_t start = 0){
    send(&v[0], v.size(), start);
  }

  void send(std::vector<T*> const& v, size_t start = 0){
    send(v[0], v.size(), start);
  }

  void fill_n(T t, size_t _size, size_t start = 0){
    if      constexpr(sizeof(T) == 64 or is_floating_point_v<T>)
      send(std::vector<T>(_size, t), start);
    else if constexpr (sizeof(T) == 1)
      KUL_GPU_ASSERT(hipMemsetD8(p + start, t, _size));
    else if constexpr (sizeof(T) == 2)
      KUL_GPU_ASSERT(hipMemsetD16(p + start, t, _size));
    else if constexpr (sizeof(T) == 4)
      KUL_GPU_ASSERT(hipMemsetD32(p + start, t, _size));
    else
      throw std::runtime_error("Unmanaged type in fill_n");
  }

  decltype(auto) operator+(size_t size){
    DeviceMem<T> view; // has no size;
    view.p = this->p + size;
    return view;
  }

  template <typename Container>
  Container& take(Container& c) const {
    KUL_GPU_ASSERT(hipMemcpy(&c[0], p, s * sizeof(T), hipMemcpyDeviceToHost));
    return c;
  }
  template <typename Container = std::vector<T>>
  Container take() const {
    Container c(s);
    return take(c);
  }
  decltype(auto) operator()() const { return take(); }
  auto& size() const { return s; }
  size_t s = 0;
  T* p = nullptr;
};

template <bool GPU>
struct ADeviceClass {};

template <>
struct ADeviceClass<true> {};

template <>
struct ADeviceClass<false> {
  ~ADeviceClass() { invalidate(); }

  void _alloc(void* ptrs, size_t size) {
    KUL_GPU_ASSERT(hipMalloc((void**)&ptr, size));
    KUL_GPU_ASSERT(hipMemcpy(ptr, ptrs, size, hipMemcpyHostToDevice));
  }

  template <typename as, typename... DevMems>
  decltype(auto) alloc(DevMems&&... mem) {
    if (ptr) throw std::runtime_error("already malloc-ed");
    auto ptrs = make_pointer_container(mem.p...);
    _alloc(&ptrs, sizeof(ptrs));
    return static_cast<as*>(ptr);
  }

  void invalidate() {
    if (ptr) {
      KUL_GPU_ASSERT(hipFree(ptr));
      ptr = nullptr;
    }
  }

  void* ptr = nullptr;
};

template <bool GPU>
struct DeviceClass : ADeviceClass<GPU> {
  template <typename T>
  using container_t = std::conditional_t<GPU, T*, kul::gpu::DeviceMem<T>>;
};

// https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/HIP-GUIDE.html#calling-global-functions
struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(size_t w, size_t h, size_t tpx, size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}

  Launcher(size_t x, size_t y, size_t z, size_t tpx, size_t tpy, size_t tpz)
      : Launcher{dim3(x / tpx, y / tpy, z / tpz), dim3(tpx, tpy, tpz)} {}

  template <typename F, typename... Args>
  void operator()(F f, Args... args) {
    hipLaunchKernelGGL(f, g, b, ds, s, args...);
  }
  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  hipStream_t s = 0;
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
#endif /* _KUL_LOG_HPP_ */
