/**
Copyright (c) 2017, Philip Deegan.
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
#ifndef _KUL_GPU_TUPLE_HPP_
#define _KUL_GPU_TUPLE_HPP_

#include "kul/gpu.hpp"

namespace kul::gpu {

template <typename T, typename SIZE = size_t>
struct Pointers {
  Pointers(T const* p_, SIZE s_) __device__ __host__ : p{p_}, s{s_} {}
  T const* p = nullptr;
  SIZE s = 0;
  auto& operator[](SIZE i) const __device__ __host__ { return p[i]; }
  auto& data() const __device__ __host__ { return p; }
  auto& begin() const __device__ __host__ { return p; }
  auto end() const __device__ __host__ { return p + s; }
  auto& size() const __device__ __host__ { return s; }
};

template <typename T, typename SIZE, bool GPU>
struct ASplitVector : kul::gpu::DeviceClass<GPU> {};

template <typename T, typename SIZE>
struct ASplitVector<T, SIZE, true> : kul::gpu::DeviceClass<true> {};

template <typename T, typename SIZE>
struct ASplitVector<T, SIZE, false> : kul::gpu::DeviceClass<false> {
  mutable kul::SplitVector<T, SIZE> base;
};

template <typename T, bool GPU, typename SIZE = size_t>
struct SplitVector : ASplitVector<T, SIZE, GPU> {
  using value_type = T;
  using Super = ASplitVector<T, SIZE, GPU>;
  using SplitVector_ = SplitVector<T, GPU, SIZE>;
  using gpu_t = SplitVector<T, true, SIZE>;

  template <typename T1>
  using container_t = typename Super::template container_t<T1>;

  /* HOST */
  template <bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  SplitVector(std::vector<SIZE>&& sizes_)
      : Super{sizes_}, sizes(sizes_), vec(Super::base.size), displs(Super::base.displs) {}

  template <bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto operator[](SIZE i) const __device__ {
    return Pointers<T, SIZE>{vec + displs[i], sizes[i]};
  }

  template <bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  decltype(auto) operator()() {
    return Super::template alloc<gpu_t>(sizes, displs, vec);
  }

  template <bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  decltype(auto) operator()() const {
    Super::base.vec = std::move(vec.take());
    return Super::base;
  }
  /* HOST */

  /* DEVICE */
  template <bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  T* data() __device__ {
    return vec;
  }

  struct iterator {
    iterator(SplitVector_* _sv) : sv(_sv) {}
    iterator operator++() {
      curr_pos += sv->sizes[curr_ptr++];
      return *this;
    }
    bool operator!=(const iterator& other) const { return curr_ptr != sv->sizes.size(); }
    Pointers<T, SIZE> operator*() const {
      return Pointers<T, SIZE>{sv->vec.data() + curr_pos, sv->sizes[curr_ptr]};
    }

    SplitVector_* sv = nullptr;
    SIZE curr_pos = 0, curr_ptr = 0;
  };

  template <bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto begin() __device__ {
    return iterator(this);
  }
  template <bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto cbegin() const __device__ {
    return iterator(this);
  }

  template <bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto end() __device__ {
    return iterator(this);
  }
  template <bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto cend() const __device__ {
    return iterator(this);
  }
  /* DEVICE */

  container_t<SIZE> sizes, displs;
  container_t<T> vec;
};

}  // end namespace kul::gpu

#endif /* _KUL_GPU_TUPLE_HPP_ */
