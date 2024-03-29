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
#ifndef _MKN_GPU_ALLOC_HPP_
#define _MKN_GPU_ALLOC_HPP_

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

template <typename T, typename Size>
void copy(T* const dst, T const* const src, Size size) {
  auto dst_p = Pointer{dst};
  auto src_p = Pointer{src};

  bool to_send = dst_p.is_device_ptr() && src_p.is_host_ptr();
  bool to_take = dst_p.is_host_ptr() && src_p.is_device_ptr();

  if (to_send)
    send(dst, src, size);
  else if (to_take)
    take(dst, src, size);
  else
    throw std::runtime_error("Unsupported operation (PR welcome)");
}

#endif /* _MKN_GPU_ALLOC_HPP_ */
