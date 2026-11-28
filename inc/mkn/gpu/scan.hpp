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
#ifndef MKN_GPU_SCAN_HPP
#define MKN_GPU_SCAN_HPP

#include "mkn/gpu.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

// device exclusive prefix sums, no thrust
//
//   auto const total = mkn::gpu::exclusive_scan(in, out, n); // out[i] = in[0] + ... + in[i - 1]
//
// in, out: n values the device can read / write (device or managed memory; host memory on the
//  CPU backend), in == out allowed. The total (in[0] + ... + in[n - 1]) is returned on the host:
//  the call's one wait, a copy of it back (the device's work up to it done). For integers: a sum
//  of floating point values depends on the grouping, which depends on n only.
// GPU, on the default stream: tiles of scan_tile values; up to scan_one_block_tiles (2) tiles one
//  block takes them in turn with a carry (one launch), past that each block sums its tile, one
//  block scans those sums into each tile's offset, each block scans its tile from its offset
//  (three launches). Within a block: each thread a run of scan_items consecutive values, a
//  shared memory (Hillis Steele) scan of the threads' sums. Scratch: one grow only buffer per
//  thread and device, no allocation once it fits. CPU: a loop.
// MKN_GPU_FN_PER_NS: exclusive_scan is the backend's (mkn::gpu::cuda::exclusive_scan, ...).

namespace MKN_GPU_NS {
namespace detail {

std::uint32_t constexpr static scan_block_size = 256;
std::uint32_t constexpr static scan_items = 8;  // consecutive values a thread
std::uint32_t constexpr static scan_tile = scan_block_size * scan_items;
std::uint32_t constexpr static scan_one_block_tiles = 2;

#if !MKN_GPU_CPU
// the block's exclusive scan of v (one value a thread) from carry, the block's total returned to
//  every thread
template <typename T>
__device__ T scan_block(T const v, T* const sums, T& carry_out) {
  std::uint32_t constexpr B = scan_block_size;
  auto const t = threadIdx.x;
  T* in = sums;
  T* out = sums + B;
  in[t] = v;
  __syncthreads();
  for (std::uint32_t o = 1; o < B; o <<= 1) {  // inclusive, double buffered
    out[t] = t >= o ? in[t] + in[t - o] : in[t];
    __syncthreads();
    T* const tmp = in;
    in = out;
    out = tmp;
  }
  carry_out = in[B - 1];
  T const exclusive = t ? in[t - 1] : T{0};
  __syncthreads();  // sums reused by the caller's next call
  return exclusive;
}

// the tile at first (n values, at most scan_tile) scanned from offset, its total returned
template <typename T>
__device__ T scan_tile_from(T const* const in, T* const out, std::size_t const first,
                            std::size_t const n, T const offset, T* const sums) {
  std::size_t const begin = first + std::size_t{threadIdx.x} * scan_items;
  std::size_t const end = std::min(begin + scan_items, first + n);
  T run{0};
  for (std::size_t i = begin; i < end; ++i) run += in[i];
  T total{0};
  T acc = offset + scan_block(run, sums, total);
  for (std::size_t i = begin; i < end; ++i) {
    T const v = in[i];  // read before the write: in == out
    out[i] = acc;
    acc += v;
  }
  return total;
}

// one block: the tiles in turn, each from the last ones' total; the total into *total
template <typename T>
__global__ void scan_one_block(T const* const in, T* const out, std::size_t const n,
                               T* const total) {
  __shared__ T sums[2 * scan_block_size];
  T carry{0};
  for (std::size_t first = 0; first < n; first += scan_tile)
    carry +=
        scan_tile_from(in, out, first, std::min<std::size_t>(scan_tile, n - first), carry, sums);
  if (threadIdx.x == 0) *total = carry;
}

// tile_sums[blockIdx.x] = the sum of block blockIdx.x's tile
template <typename T>
__global__ void scan_tile_sums(T const* const in, std::size_t const n, T* const tile_sums) {
  __shared__ T sums[2 * scan_block_size];
  std::size_t const first = std::size_t{blockIdx.x} * scan_tile;
  std::size_t const begin = first + std::size_t{threadIdx.x} * scan_items;
  std::size_t const end = std::min(begin + scan_items, std::min(first + scan_tile, n));
  T run{0};
  for (std::size_t i = begin; i < end; ++i) run += in[i];
  T total{0};
  scan_block(run, sums, total);
  if (threadIdx.x == 0) tile_sums[blockIdx.x] = total;
}

// each block's tile scanned from its offset (tile_sums scanned)
template <typename T>
__global__ void scan_tiles(T const* const in, T* const out, std::size_t const n,
                           T const* const offsets) {
  __shared__ T sums[2 * scan_block_size];
  std::size_t const first = std::size_t{blockIdx.x} * scan_tile;
  scan_tile_from(in, out, first, std::min<std::size_t>(scan_tile, n - first), offsets[blockIdx.x],
                 sums);
}

// per thread and device scratch of at least n T's, grown only: an allocation belongs to the
//  device current when it was made
template <typename T>
T* scan_scratch(std::size_t const n) {
  struct Scratch {
    T* p = nullptr;
    std::size_t size = 0;
    ~Scratch() {
      if (p) destroy(p);
    }
  };
  thread_local std::vector<std::unique_ptr<Scratch>> per_device;
  auto const dev = getDevice();
  if (per_device.size() <= dev) per_device.resize(dev + 1);
  if (!per_device[dev]) per_device[dev] = std::make_unique<Scratch>();
  auto& scratch = *per_device[dev];
  if (n > scratch.size) {
    if (scratch.p) destroy(scratch.p);
    scratch.size = n + n / 2;
    alloc(scratch.p, scratch.size);
  }
  return scratch.p;
}
#endif  // !MKN_GPU_CPU

}  // namespace detail

template <typename T>
T exclusive_scan(T const* const in, T* const out, std::size_t const n) {
  static_assert(std::is_arithmetic_v<T>, "mkn::gpu::exclusive_scan: arithmetic values");
  if (n == 0) return T{0};
#if MKN_GPU_CPU
  T acc{0};
  for (std::size_t i = 0; i < n; ++i) {
    T const v = in[i];
    out[i] = acc;
    acc += v;
  }
  return acc;
#else
  std::uint32_t constexpr B = detail::scan_block_size;
  std::size_t const tiles = (n + detail::scan_tile - 1) / detail::scan_tile;
  T* const scratch = detail::scan_scratch<T>(tiles + 1);  // [tiles]: the total
  T* const total = scratch + tiles;
  if (tiles <= detail::scan_one_block_tiles) {
    Launcher one{dim3(1), dim3(B)};
    launch<false>(detail::scan_one_block<T>, one.g, one.b, 0, one.s, in, out, n, total);
  } else {
    Launcher all{dim3(static_cast<std::uint32_t>(tiles)), dim3(B)}, one{dim3(1), dim3(B)};
    launch<false>(detail::scan_tile_sums<T>, all.g, all.b, 0, all.s, in, n, scratch);
    launch<false>(detail::scan_one_block<T>, one.g, one.b, 0, one.s, scratch, scratch, tiles,
                  total);
    launch<false>(detail::scan_tiles<T>, all.g, all.b, 0, all.s, in, out, n, scratch);
  }
  T result{0};
  take(total, &result, 1);
  return result;
#endif
}

}  // namespace MKN_GPU_NS

#endif /* MKN_GPU_SCAN_HPP */
