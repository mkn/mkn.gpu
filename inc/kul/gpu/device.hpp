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

template <typename T>
struct DeviceMem {
  using value_type = T;

  DeviceMem() {}
  DeviceMem(std::size_t _s) : s{_s}, owned{true} {
    if (s) alloc(p, s);
  }

  DeviceMem(DeviceMem const&) = delete;
  DeviceMem(DeviceMem&&) = delete;
  auto& operator=(DeviceMem const&) = delete;
  auto& operator=(DeviceMem&&) = delete;

  DeviceMem(T const* t, std::size_t _s) : DeviceMem{_s} { send(t, _s); }

  template <typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  DeviceMem(C const& c) : DeviceMem{c.data(), c.size()} {}

  template <typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  DeviceMem(C&& c) : DeviceMem{c.data(), c.size()} {}

  ~DeviceMem() {
    if (p && s && owned) destroy(p);
  }

  void send(T const* t, std::size_t _size = 1, std::size_t start = 0) {
    KUL_GPU_NS::send(p, t, _size, start);
  }

  template <typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  void send(C const& c, std::size_t start = 0) {
    send(c.data(), c.size(), start);
  }

  void fill_n(T t, std::size_t _size, std::size_t start = 0) {
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

  void take(T* to, std::size_t size) { KUL_GPU_NS::take(p, to, size); }
  void take(T* to) { KUL_GPU_NS::take(p, to, s); }

  template <typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  C& take(C& c) {
    take(c.data(), c.size());
    return c;
  }

  template <typename C = std::vector<T>, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
  C take() {
    C c(s);
    return take(c);
  }

  auto operator()(T* to) { return take(to); }
  auto operator()() { return take(); }

  auto& size() const { return s; }

  std::size_t s = 0;
  T* p = nullptr;
  bool owned = false;
};

template <typename T>
struct AsioDeviceMem {
  AsioDeviceMem(AsioDeviceMem const&) = delete;
  AsioDeviceMem(AsioDeviceMem&&) = delete;
  auto& operator=(AsioDeviceMem const&) = delete;
  auto& operator=(AsioDeviceMem&&) = delete;

  AsioDeviceMem(std::size_t _s = 0) : s{_s} {
    assert(p == nullptr);
    if (s) KUL_GPU_NS::alloc(p, s);
    assert(p != nullptr);
  }

  ~AsioDeviceMem() {
    if (p && s) KUL_GPU_NS::destroy(p);
    p = nullptr;
  }

  void send(Stream& stream, T* t, std::size_t _size = 1, std::size_t start = 0) {
    assert(p != nullptr);
    KUL_GPU_NS::send_async(p, t, stream, _size, start);
  }

  template <typename Span>
  void take(Stream& stream, Span& span, std::size_t start) {
    assert(p != nullptr);
    assert(span.size() + start <= s);
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
    if (!ptr) KUL_GPU_NS::alloc(ptr, size);
    KUL_GPU_NS::send(ptr, ptrs, size);
  }

  template <typename as, typename... DevMems>
  auto alloc(DevMems&... mem) {
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
  template <typename T>
  using container_t = std::conditional_t<GPU, T*, DeviceMem<T>>;
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

template <typename T, std::size_t size_>
struct HostArray {
  using value_type = T;

  HostArray(HostArray const&) = delete;

  HostArray() {
    if constexpr (size_ > 0) KUL_GPU_NS::alloc_host(p, size_);
  }

  ~HostArray() {
    if constexpr (size_ > 0)
      if (p) KUL_GPU_NS::destroy_host(p);
  }

  auto& operator[](std::size_t idx) {
    assert(idx < size_);
    return p[idx];
  }
  auto& operator[](std::size_t idx) const {
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

template <typename T>
struct HostMem {
  using value_type = T;

  HostMem(HostMem& that) : p{that.p}, size_{that.size_} { that.p = nullptr; }

  HostMem(std::size_t _size) : size_{_size} {
    if (size_ > 0) KUL_GPU_NS::alloc_host(p, size_);
  }

  HostMem(T* const data, std::size_t _size) {
    size_ = _size;
    assert(size_ > 0);
    KUL_GPU_NS::alloc_host(p, size_);
    std::copy(data, data + _size, p);
  }

  template <template <typename> typename C, std::enable_if_t<kul::is_span_like_v<C<T>>, bool> = 0>
  HostMem(C<T> const& c) : HostMem{c.data(), c.size()} {}

  ~HostMem() {
    if (size_ > 0)
      if (p) KUL_GPU_NS::destroy_host(p);
  }

  auto& operator[](std::size_t idx) {
    assert(idx < size_);
    return p[idx];
  }

  auto& front() const { return p[0]; }
  auto& back() const { return p[size_ - 1]; }

  auto begin() { return p; }
  auto begin() const { return p; }
  auto end() { return p + size_; }
  auto end() const { return p + size_; }
  auto* data() { return p; }
  auto* data() const { return p; }
  auto size() const { return size_; }

  T* p = nullptr;
  std::size_t size_;
};

template <typename T>
struct is_host_mem : std::false_type {};
template <typename T>
struct is_host_mem<HostMem<T>> : std::true_type {};
template <typename T>
inline constexpr auto is_host_mem_v = is_host_mem<T>::value;

namespace {

template <typename T>
struct is_std_unique_ptr : std::false_type {};
template <typename T>
struct is_std_unique_ptr<std::unique_ptr<T>> : std::true_type {};
template <typename T>
inline constexpr auto is_std_unique_ptr_v = is_std_unique_ptr<T>::value;

template <typename T>
struct is_std_shared_ptr : std::false_type {};
template <typename T>
struct is_std_shared_ptr<std::shared_ptr<T>> : std::true_type {};
template <typename T>
inline constexpr auto is_std_shared_ptr_v = is_std_shared_ptr<T>::value;

template <typename T>
struct is_ref_devmen : std::false_type {};
template <typename T>
struct is_ref_devmen<std::reference_wrapper<DeviceMem<T>>> : std::true_type {};
template <typename T>
inline constexpr auto is_ref_devmen_v = is_ref_devmen<T>::value;
    
template <typename T>
struct is_ref_wrap : std::false_type {};
template <typename T>
struct is_ref_wrap<std::reference_wrapper<T>> : std::true_type {};
template <typename T>
inline constexpr auto is_ref_wrap_v = is_ref_wrap<T>::value; 

template <typename T>
struct is_asio_mem : std::false_type {};
template <typename T>
struct is_asio_mem<AsioDeviceMem<T>> : std::true_type {};
template <typename T>
inline constexpr auto is_asio_mem_v = is_asio_mem<T>::value;
    

    
    

template <typename T0>
auto handle_input(T0& t) {
  using T = std::decay_t<T0>;
  if constexpr (is_device_mem_v<T>) {
    return std::ref(t);
  } else if constexpr (std::is_base_of_v<DeviceClass<false>, T>) {
    return std::ref(t);
  } else if constexpr (kul::is_span_like_v<T>) {
    return std::make_shared<DeviceMem<typename T::value_type>>(t);
  } else {
    return std::make_shared<DeviceMem<T>>(&t, 1);
  }
}

template <std::size_t... I, typename... Args>
auto handle_inputs(std::tuple<Args&...>& tup, std::index_sequence<I...>) {
  return std::make_tuple(handle_input(std::get<I>(tup))...);
}

template<typename T>
constexpr bool t_is_lval(){
  if constexpr (is_ref_wrap_v<T>)    
    return (std::is_base_of_v<DeviceClass<false>, typename T::type>);  
  return  (std::is_base_of_v<DeviceClass<false>, T>);
}

template<typename T0, std::enable_if_t<t_is_lval<T0>(), int> = 0>
auto replace(T0& t){
  using T = std::decay_t<T0>;
  KLOG(TRC) << typeid(t).name();
  if constexpr (is_ref_wrap_v<T>)
      if constexpr(std::is_base_of_v<DeviceClass<false>, typename T::type>)
          return t()();   
  if constexpr (std::is_base_of_v<DeviceClass<false>, T>) 
    return t();
}

template<typename T0, std::enable_if_t<!t_is_lval<T0>(), int> = 0>
auto& replace(T0& t) {
  using T = std::decay_t<T0>;
  KLOG(TRC) << typeid(t).name();
  if constexpr (is_ref_devmen_v<T>) {
    return t().p;
  } else if constexpr (is_asio_mem_v<T>) {
    assert(t.s > 0);
    assert(t.p != nullptr);
    return t.p;
  } else if constexpr (is_device_mem_v<T>) {
    assert(t.s > 0);
    assert(t.p != nullptr);
    return t.p;
  } else if constexpr (std::is_base_of_v<DeviceClass<false>, T>) {
//     throw std::runtime_error("NO");
//     static_assert(!std::is_base_of_v<DeviceClass<false>, T>);
//     return t();
  } else if constexpr (std::is_base_of_v<DeviceClass<true>, T>) {
    return t;
  } else if constexpr (is_std_unique_ptr_v<T>) {
    static_assert(is_device_mem_v<typename T::element_type>);
    return t->p;
  } else if constexpr (is_std_shared_ptr_v<T>) {
    static_assert(is_device_mem_v<typename T::element_type>);
    return t->p;
  } else {
    return t;
  }
}

template <std::size_t... I, typename... Args>
auto devmem_replace(std::tuple<Args&...>&& tup, std::index_sequence<I...>) {
  KLOG(TRC);
  return std::make_tuple(replace(std::get<I>(tup))...);
}

} /* namespace */

#endif /* _KUL_GPU_DEVICE_HPP_ */
