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
#ifndef _KUL_GPU_DEVICE_HPP_
#define _KUL_GPU_DEVICE_HPP_

template <typename T, typename SIZE = uint32_t>
struct DeviceMem {
  using value_type = T;

  DeviceMem() {}
  DeviceMem(SIZE _s) : s{_s}, owned{true} {
    if (s) alloc(p, s);
  }

  DeviceMem(T const* t, SIZE _s) : DeviceMem{_s} { send(t, _s); }

  template <typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  DeviceMem(C const& c) : DeviceMem{c.data(), static_cast<SIZE>(c.size())} {}

  template <typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  DeviceMem(C&& c) : DeviceMem{c.data(), static_cast<SIZE>(c.size())} {}

  ~DeviceMem() {
    if (p && s && owned) destroy(p);
  }

  void send(T const* t, SIZE _size = 1, SIZE start = 0) { KUL_GPU_NS::send(p, t, _size, start); }

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
    KUL_GPU_NS::take(p, &c[0], s);
    return c;
  }

  template <typename Container = std::vector<T>>
  Container take() const {
    Container c(s);
    return take(c);
  }

  auto operator()() const { return take(); }

  auto& size() const { return s; }

  SIZE s = 0;
  T* p = nullptr;
  bool owned = false;
};

template <typename T, typename SIZE = uint32_t>
struct AsioDeviceMem {
  AsioDeviceMem(std::size_t _s = 0) : s{_s} {
    if (s) KUL_GPU_NS::alloc(p, s);
  }

  ~AsioDeviceMem() {
    if (p && s) KUL_GPU_NS::destroy(p);
  }

  void send(Stream& stream, T const* t, SIZE _size = 1, SIZE start = 0) {
    KUL_GPU_NS::send_async(p, t, stream, _size, start);
  }

  template <typename Span>
  void take(Stream& stream, Span& span, std::size_t start) {
    KUL_GPU_NS::take_async(p, span, stream, start);
  }

  auto& size() const { return s; }

  std::size_t s = 0;
  T* p = nullptr;
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
    KUL_GPU_NS::alloc(ptr, size);
    KUL_GPU_NS::send(ptr, ptrs, size);
  }

  template <typename as, typename... DevMems>
  auto alloc(DevMems&... mem) {
    if (ptr) throw std::runtime_error("already malloc-ed");
    auto ptrs = make_pointer_container(mem.p...);
    static_assert(sizeof(as) == sizeof(ptrs), "Class cast type size mismatch");

    _alloc(&ptrs, sizeof(ptrs));
    return static_cast<as*>(ptr);
  }

  void invalidate() {
    if (ptr) {
      destroy(ptr);
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

template <typename T, typename V>
void fill_n(DeviceMem<T>& p, size_t size, V val) {
  p.fill_n(val, size);
}

template <typename T, typename V>
void fill_n(DeviceMem<T>&& p, size_t size, V val) {
  fill_n(p, size, val);
}

template <typename T, typename SIZE, SIZE size_>
struct HostArrayBase {
  using value_type = T;

  HostArrayBase(HostArrayBase const&) = delete;

  HostArrayBase() {
    if constexpr (size_ > 0) KUL_GPU_NS::alloc_host(p, size_);
  }

  ~HostArrayBase() {
    if constexpr (size_ > 0)
      if (p) KUL_GPU_NS::destroy_host(p);
  }

  auto& operator[](SIZE idx) {
    assert(idx < size_);
    return p[idx];
  }
  auto& operator[](SIZE idx) const {
    assert(idx < size_);
    return p[idx];
  }

  auto begin() { return p; }
  auto begin() const { return p; }
  auto end() { return p + size_; }
  auto end() const { return p + size_; }
  auto* data() { return p; }
  auto* data() const { return p; }

  constexpr auto size() const { return size_; }

  T* p = nullptr;
};

template <typename T, std::uint32_t size>
using HostArray = HostArrayBase<T, std::uint32_t, size>;

template <typename T, std::size_t size>
using BigHostArray = HostArrayBase<T, std::size_t, size>;

template <typename T, typename SIZE_t = std::uint32_t>
struct HostMem {
  using value_type = T;
  using SIZE = SIZE_t;

  HostMem(HostMem& that) : p{that.p}, size_{that.size_} { that.p = nullptr; }

  HostMem(SIZE _size) : size_{_size} {
    if (size_ > 0) KUL_GPU_NS::alloc_host(p, size_);
  }

  ~HostMem() {
    if (size_ > 0)
      if (p) KUL_GPU_NS::destroy_host(p);
  }

  auto& operator[](SIZE idx) {
    assert(idx < size_);
    return p[idx];
  }

  auto begin() { return p; }
  auto begin() const { return p; }
  auto end() { return p + size_; }
  auto end() const { return p + size_; }
  auto* data() { return p; }
  auto* data() const { return p; }
  auto size() const { return size_; }

  T* p = nullptr;
  SIZE size_;
};

template <typename T, typename SIZE_t = std::uint32_t>
struct is_host_mem : std::false_type {};

template <typename T, typename SIZE_t>
struct is_host_mem<HostMem<T, SIZE_t>> : std::true_type {};

template <typename T, typename SIZE_t>
inline constexpr auto is_host_mem_v = is_host_mem<T, SIZE_t>::value;

namespace {

template <typename T>
auto replace(T& t) {
  if constexpr (is_device_mem_v<T>)
    return t.p;
  else if constexpr (kul::is_std_unique_ptr_v<T>) {
    static_assert(is_device_mem_v<typename T::element_type>);
    return t->p;
  } else if constexpr (kul::is_std_shared_ptr_v<T>) {
    static_assert(is_device_mem_v<typename T::element_type>);
    return t->p;
  } else
    return t;
}

template <std::size_t... I, typename... Args>
auto devmem_replace(std::tuple<Args&...>&& tup, std::index_sequence<I...>) {
  return std::make_tuple(replace(std::get<I>(tup))...);
}

} /* namespace */

#endif /* _KUL_GPU_DEVICE_HPP_ */
