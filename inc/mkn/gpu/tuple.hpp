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
#ifndef _MKN_GPU_TUPLE_HPP_
#define _MKN_GPU_TUPLE_HPP_

// #include "mkn/gpu.hpp"
#include "mkn/kul/tuple.hpp"

namespace mkn::gpu {

template <typename T, typename SIZE = std::size_t>
struct Span {
  using value_type = T;

  Span(T* ptr_, SIZE s_) __device__ __host__ : ptr{ptr_}, s{s_} {}

  auto& operator[](SIZE i) __device__ __host__ { return ptr[i]; }
  auto const& operator[](SIZE i) const __device__ __host__ { return ptr[i]; }
  auto data() __device__ __host__ { return ptr; }
  auto data() const __device__ __host__ { return ptr; }
  auto begin() __device__ __host__ { return ptr; }
  auto cbegin() const __device__ __host__ { return ptr; }
  auto end() __device__ __host__ { return ptr + s; }
  auto cend() const __device__ __host__ { return ptr + s; }
  SIZE const& size() const __device__ __host__ { return s; }

  auto& front() const __device__ __host__ {
    assert(s > 0);
    return ptr[0];
  }

  auto& back() const __device__ __host__ {
    assert(s > 0);
    return ptr[s - 1];
  }

  T* ptr = nullptr;
  SIZE s = 0;
};

template <typename T, typename SIZE, bool GPU>
struct ASpanSet : mkn::gpu::DeviceClass<GPU> {};

template <typename T, typename SIZE>
struct ASpanSet<T, SIZE, true> : mkn::gpu::DeviceClass<true> {};

template <typename T, typename SIZE>
struct ASpanSet<T, SIZE, false> : mkn::gpu::DeviceClass<false> {
  mkn::kul::SpanSet<T, SIZE> base;
};

template <typename T, typename SIZE = size_t, bool GPU = false>
struct SpanSet : ASpanSet<T, SIZE, GPU> {
  using value_type = T;
  using Super = ASpanSet<T, SIZE, GPU>;
  using SpanSet_ = SpanSet<T, SIZE, GPU>;
  using gpu_t = SpanSet<T, SIZE, true>;

  template <typename T1>
  using container_t = typename Super::template container_t<T1>;

  /* HOST */
  template <bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  SpanSet(std::vector<SIZE>&& sizes_)
      : Super{sizes_}, sizes(sizes_), vec(Super::base.size), displs(Super::base.displs) {}

  template <bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto operator[](SIZE i) __device__ {
    return Span<T, SIZE>{vec + displs[i], sizes[i]};
  }

  template <bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto operator[](SIZE i) const __device__ {
    return Span<T, SIZE>{vec + displs[i], sizes[i]};
  }

  template <bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  auto send() {
    return Super::template alloc<gpu_t>(sizes, displs, vec);
  }

  template <bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  auto& take() {
    Super::base.vec = std::move(vec.take());
    return Super::base;
  }
  /* HOST */

  /* DEVICE */
  struct iterator {
    iterator(SpanSet_* _sv) __device__ : sv(_sv) {}
    iterator operator++() __device__ {
      curr_pos += sv->sizes[curr_ptr++];
      return *this;
    }
    bool operator!=(iterator const& /*other*/) const __device__ {
      return curr_ptr != sv->sizes.size();
    }
    Span<T, SIZE> operator*() const {
      return Span<T, SIZE>{sv->vec.data() + curr_pos, sv->sizes[curr_ptr]};
    }

    SpanSet_* sv = nullptr;
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

}  // end namespace mkn::gpu

#endif /* _MKN_GPU_TUPLE_HPP_ */
