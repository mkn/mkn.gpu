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
#endif  // KUL_GPU_FN_PER_NS

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

template <typename T, typename SIZE = uint32_t>
struct DeviceMem {

  DeviceMem() {}
  DeviceMem(SIZE _s) : s{_s}, owned{true} {
    SIZE alloc_bytes = s * sizeof(T);
    KLOG(OTH) << "GPU alloced: " << alloc_bytes;
    if (s) KUL_GPU_ASSERT(hipMalloc((void**)&p, alloc_bytes));
  }

  DeviceMem(T const* const t, SIZE _s) : DeviceMem{_s} { send(t, _s); }
  template <typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  DeviceMem(C c) : DeviceMem{c.data(), static_cast<SIZE>(c.size())} {}

  ~DeviceMem() {
    if (p && s && owned) KUL_GPU_ASSERT(hipFree(p));
  }

  void send(T const* const t, SIZE _size = 1, SIZE start = 0) {
    KUL_GPU_ASSERT(hipMemcpy(p + start, t, _size * sizeof(T), hipMemcpyHostToDevice));
  }
  template <typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  void send(C c, SIZE start = 0) {
    send(c.data(), c.size(), start);
  }

  void fill_n(T t, SIZE _size, SIZE start = 0) {
    // TODO - improve with memSet style
    assert(_size + start <= s);
    send(std::vector<T>(_size, t), start);
  }

  DeviceMem<T> operator+(size_t size) {
    DeviceMem<T> view;
    view.p = this->p + size;
    view.s = this->s - size;
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

  SIZE s = 0;
  T* p = nullptr;
  bool owned = false;
};

template <typename T>
struct is_device_mem : std::false_type {};

template <typename T>
struct is_device_mem<DeviceMem<T>> : std::true_type {};

template <typename T>
inline constexpr auto is_device_mem_v = is_device_mem<T>::value;

template <bool GPU>
struct ADeviceClass {};

template <>
struct ADeviceClass<true> {};

template <>
struct ADeviceClass<false> {
  ~ADeviceClass() { invalidate(); }

  void _alloc(void* ptrs, uint8_t size) {
    KUL_GPU_ASSERT(hipMalloc((void**)&ptr, size));
    KUL_GPU_ASSERT(hipMemcpy(ptr, ptrs, size, hipMemcpyHostToDevice));
  }

  template <typename as, typename... DevMems>
  decltype(auto) alloc(DevMems&... mem) {
    if (ptr) throw std::runtime_error("already malloc-ed");
    auto ptrs = make_pointer_container(mem.p...);
    static_assert(sizeof(as) == sizeof(ptrs), "Class cast type size mismatch");

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

template <bool GPU = false>
struct DeviceClass : ADeviceClass<GPU> {
  template <typename T, typename SIZE = uint32_t>
  using container_t = std::conditional_t<GPU, T*, DeviceMem<T, SIZE>>;
};

using HostClass = DeviceClass<false>;

namespace {

template <typename T>
decltype(auto) replace(T& t) {
  if constexpr (is_device_mem_v<T>)
    return t.p;
  else
    return t;
}

template <std::size_t... IS, typename... Args>
decltype(auto) devmem_replace(std::tuple<Args&...>&& tup, std::index_sequence<IS...>) {
  return std::make_tuple(replace(std::get<IS>(tup))...);
}

} /* namespace */

void sync() { KUL_GPU_ASSERT(hipDeviceSynchronize()); }

// https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/HIP-GUIDE.html#calling-global-functions
struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(size_t w, size_t h, size_t tpx, size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}
  Launcher(size_t x, size_t y, size_t z, size_t tpx, size_t tpy, size_t tpz)
      : Launcher{dim3(x / tpx, y / tpy, z / tpz), dim3(tpx, tpy, tpz)} {}

  template <typename F, typename... Args>
  void operator()(F f, Args&&... args) {
    sync();
    std::apply([&](auto&&... params) { hipLaunchKernelGGL(f, g, b, ds, s, params...); },
               devmem_replace(std::forward_as_tuple(args...),
                              std::make_index_sequence<sizeof...(Args)>()));
  }
  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  hipStream_t s = 0;
};

template <typename T, typename V>
void fill_n(DeviceMem<T>& p, size_t size, V val) {
  p.fill_n(val, size);
}

template <typename T, typename V>
void fill_n(DeviceMem<T>&& p, size_t size, V val) {
  fill_n(p, size, val);
}

#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
} /* namespace hip */
#endif  // KUL_GPU_FN_PER_NS
} /* namespace kul::gpu */

#undef KUL_GPU_ASSERT
#endif /* _KUL_GPU_ROCM_HPP_ */
