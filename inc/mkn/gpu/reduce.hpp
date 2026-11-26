/**
Copyright (c) 2026, Philip Deegan.
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
#ifndef MKN_GPU_REDUCE_HPP
#define MKN_GPU_REDUCE_HPP

#include "mkn/gpu.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

// device reductions: one value from a buffer, back on the host, no thrust
//
//   double const total = mkn::gpu::reduce(ptr, n, 0., mkn::gpu::reduce_op::Sum{});
//   float const least  = mkn::gpu::reduce(ptr, n, inf, mkn::gpu::reduce_op::Min{});
//
// ptr: n values the device can read (device or managed memory; host memory on the CPU
//  backend). The result is op(init, op over the values), init once.
// GPU: two launches on the default stream, synchronous: blocks of `block_size` threads each
//  fold a grid stride share of the values, then a shared memory tree per block, one partial per
//  block (at most `max_blocks`); one block folds the partials. The grouping depends on n only,
//  not on the device or timing: floating point sums are the same run to run (unlike atomics),
//  though not the sequential sum's bits. CPU: a loop over the values, in order.
// An op: a device and host callable op(a, b), associative and commutative (values are grouped by
//  thread and block, not taken in order), and identity<T>(): op(identity, x) == op(x, identity)
//  == x (each thread starts from it, threads without values feed it to the tree, init is applied
//  once).
// MKN_GPU_FN_PER_NS: reduce is the backend's (mkn::gpu::cuda::reduce, ...), reduce_op is shared.

namespace mkn::gpu {
namespace reduce_op {

struct Sum {
  template <typename T>
  __host__ __device__ static constexpr T identity() {
    return T{0};
  }
  template <typename T>
  __host__ __device__ T operator()(T const a, T const b) const {
    return a + b;
  }
};

struct Min {
  template <typename T>
  __host__ __device__ static constexpr T identity() {
    if constexpr (std::numeric_limits<T>::has_infinity)
      return std::numeric_limits<T>::infinity();
    else
      return std::numeric_limits<T>::max();
  }
  template <typename T>
  __host__ __device__ T operator()(T const a, T const b) const {
    return b < a ? b : a;
  }
};

struct Max {
  template <typename T>
  __host__ __device__ static constexpr T identity() {
    if constexpr (std::numeric_limits<T>::has_infinity)
      return -std::numeric_limits<T>::infinity();
    else
      return std::numeric_limits<T>::lowest();
  }
  template <typename T>
  __host__ __device__ T operator()(T const a, T const b) const {
    return a < b ? b : a;
  }
};

}  // namespace reduce_op
}  // namespace mkn::gpu

namespace MKN_GPU_NS {
namespace detail {

std::uint32_t constexpr static reduce_block_size = 256;  // a power of 2 (the tree halves it)
std::uint32_t constexpr static reduce_max_blocks = 1024;

#if !MKN_GPU_CPU
// out[blockIdx.x] = op over in[i] for i = blockIdx.x * B + threadIdx.x + k * gridDim.x * B < n
template <typename T, typename Op>
__global__ void reduce_blocks(T const* const in, std::size_t const n, Op const op, T* const out) {
  std::uint32_t constexpr B = reduce_block_size;
  __shared__ T tree[B];
  T acc = Op::template identity<T>();
  std::size_t const stride = static_cast<std::size_t>(gridDim.x) * B;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * B + threadIdx.x; i < n; i += stride)
    acc = op(acc, in[i]);
  tree[threadIdx.x] = acc;
  __syncthreads();
  for (std::uint32_t w = B / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) tree[threadIdx.x] = op(tree[threadIdx.x], tree[threadIdx.x + w]);
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = tree[0];
}
#endif  // !MKN_GPU_CPU

}  // namespace detail

template <typename T, typename Op>
T reduce(T const* const ptr, std::size_t const n, T const init, Op const op) {
  static_assert(std::is_trivially_copyable_v<T>, "mkn::gpu::reduce: trivially copyable values");
  if (n == 0) return init;
#if MKN_GPU_CPU
  T acc = Op::template identity<T>();
  for (std::size_t i = 0; i < n; ++i) acc = op(acc, ptr[i]);
  return op(init, acc);
#else
  std::uint32_t constexpr B = detail::reduce_block_size;
  auto const blocks =
      static_cast<std::uint32_t>(std::min<std::size_t>((n + B - 1) / B, detail::reduce_max_blocks));
  DeviceMem<T> partials(blocks), total(1);
  Launcher first{dim3(blocks), dim3(B)}, second{dim3(1), dim3(B)};
  launch(detail::reduce_blocks<T, Op>, first.g, first.b, 0, first.s, ptr, n, op, partials.data());
  launch(detail::reduce_blocks<T, Op>, second.g, second.b, 0, second.s, partials.data(),
         std::size_t{blocks}, op, total.data());
  T result;
  total.take(&result, 1);
  return op(init, result);
#endif
}

template <typename T, typename Op>
T reduce(DeviceMem<T> const& mem, T const init, Op const op) {
  return reduce(mem.data(), mem.size(), init, op);
}

}  // namespace MKN_GPU_NS

#endif /* MKN_GPU_REDUCE_HPP */
