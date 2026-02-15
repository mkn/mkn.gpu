/**
Copyright (c) 2024, Philip Deegan.
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
#ifndef _MKN_GPU_ALLOC_HPP_
#define _MKN_GPU_ALLOC_HPP_

template <typename T, std::int32_t alignment>
class MknGPUAllocator {
  using This = MknGPUAllocator<T, alignment>;

 public:
  using pointer = T*;
  using reference = T&;
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    using other = MknGPUAllocator<U, alignment>;
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

template <typename T, std::int32_t alignment = 32>
class NoConstructAllocator : public MknGPUAllocator<T, alignment> {
 public:
  template <typename U>
  struct rebind {
    using other = NoConstructAllocator<U, alignment>;
  };

  template <typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    ::new ((void*)ptr) U(std::forward<Args>(args)...);
  }
  template <typename U>
  void construct(U* /*ptr*/) noexcept(std::is_nothrow_default_constructible<U>::value) {}
};

template <typename T, std::int32_t align>
std::vector<T, MknGPUAllocator<T, align>>& as_super(
    std::vector<T, NoConstructAllocator<T, align>>& v) {
  return *reinterpret_cast<std::vector<T, MknGPUAllocator<T, align>>*>(&v);
}

template <typename T, std::int32_t alignment = 32>
class ManagedAllocator : public MknGPUAllocator<T, alignment> {
 public:
  template <typename U>
  struct rebind {
    using other = ManagedAllocator<U, alignment>;
  };
};

template <typename T, std::int32_t align>
std::vector<T, MknGPUAllocator<T, align>>& as_super(std::vector<T, ManagedAllocator<T, align>>& v) {
  return *reinterpret_cast<std::vector<T, MknGPUAllocator<T, align>>*>(&v);
}

template <typename T0, typename T1, typename Size>
void copy(T0* dst, T1* src, Size const size) {
  assert(dst and src);

  Pointer src_p{src};
  Pointer dst_p{dst};

  auto to_send = [&]() { return dst_p.is_device_ptr() && src_p.is_host_ptr(); };
  auto to_take = [&]() { return dst_p.is_host_ptr() && src_p.is_device_ptr(); };
  auto on_host = [&]() { return dst_p.is_host_ptr() && src_p.is_host_ptr(); };
  auto on_device = [&]() { return dst_p.is_device_ptr() && src_p.is_device_ptr(); };

  if (on_host())
    std::copy(src, src + size, dst);
  else if (on_device())
    copy_on_device(dst, src, size);
  else if (to_send())
    send(dst, src, size);
  else if (to_take())
    take(src, dst, size);
  else
    throw std::runtime_error("Unsupported operation (PR welcome)");
}

template <typename T, std::int32_t align>
auto& reserve(std::vector<T, NoConstructAllocator<T, align>>& v, std::size_t const& s,
              bool mem_copy = true) {
  if (s <= v.capacity()) {
    v.reserve(s);
    return v;
  }
  std::vector<T, NoConstructAllocator<T, align>> cpy(NoConstructAllocator<T, align>{});
  cpy.reserve(s);
  cpy.resize(v.size());
  if (mem_copy and v.size()) copy(cpy.data(), v.data(), v.size());
  v = std::move(cpy);
  return v;
}

template <typename T, std::int32_t align>
auto& resize(std::vector<T, NoConstructAllocator<T, align>>& v, std::size_t const& s,
             bool mem_copy = true) {
  if (s <= v.capacity()) {
    v.resize(s);
    return v;
  }
  std::vector<T, NoConstructAllocator<T, align>> cpy(NoConstructAllocator<T, align>{});
  cpy.resize(s);
  if (mem_copy and v.size()) copy(cpy.data(), v.data(), v.size());
  v = std::move(cpy);
  return v;
}

#endif /* _MKN_GPU_ALLOC_HPP_ */
