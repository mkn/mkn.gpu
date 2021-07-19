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
#ifndef _KUL_GPU_CUDA_ASIO_HPP_
#define _KUL_GPU_CUDA_ASIO_HPP_

#include "kul/span.hpp"
#include "kul/gpu/cuda.hpp"
#include "kul/gpu/tuple.hpp"

#define KUL_GPU_ASSERT(x) (KASSERT((x) == cudaSuccess))

// review
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu

namespace kul::gpu {
#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
namespace cuda {
#endif  // KUL_GPU_FN_PER_NS

namespace asio {

template <typename SIZE = uint32_t /*max 4294967296*/>
__device__ SIZE idx() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

template <typename SIZE>
__device__ SIZE idx(SIZE offset) {
  return (threadIdx.x + blockIdx.x * blockDim.x) + offset;
}

template <typename T>
auto handle_input(T const& t) {
  if constexpr (is_device_mem_v<T>)
    return t;
  else if constexpr (std::is_base_of_v<HostClass, T>)
    return const_cast<T&>(t)();
  else if constexpr (kul::is_span_like_v<T>)
    return std::make_shared<DeviceMem<typename T::value_type>>(t);
  else
    return std::make_shared<DeviceMem<T>>(&t, 1);
}

template <std::size_t... I, typename... Args>
auto handle_inputs(std::tuple<Args&...>& tup, std::index_sequence<I...>) {
  return std::make_tuple(handle_input(std::get<I>(tup))...);
}

class Launcher {
 private:
  Launcher(dim3 _g, dim3 _b, std::size_t batch_size_ = 1) : batch_size{batch_size_}, g{_g}, b{_b} {}

 public:
  Launcher(size_t tpx, std::size_t batch_size_ = 1) : Launcher{dim3(), dim3(tpx), batch_size_} {}

  template <typename F, typename Async, typename... Args>
  auto operator()(F& f, Async& async, Args&... args) {
    auto tuple = std::forward_as_tuple(args...);
    auto constexpr tuple_size = std::tuple_size_v<decltype(tuple)>;

    auto rest = handle_inputs(tuple, std::make_index_sequence<tuple_size>());
    using Tuple = decltype(rest);
    using Batch_t = Batch<Async, Tuple>;
    auto batch = std::make_unique<Batch_t>(batch_size, async, std::move(rest));
    auto& batch_r = (*batch);

    // Batch batch{batch_size, async, handle_inputs(tuple, std::make_index_sequence<tuple_size>())};
    auto& _asio = batch->_asio;
    auto& streams = batch->streams;
    auto& streamSize = batch->streamSize;

    auto _g = g;
    assert(g.x == _g.x);
    _g.x = streamSize / b.x;

    for (std::size_t i = 0; i < streams.size(); ++i) {
      batch_r(i);
      std::apply(
          [&](auto&&... params) {
            KUL_GPU_NS::launch(f, _g, b, ds, streams[i](), i * streamSize, params...);
          },
          batch_r());
    }

    batch->async_back();
    return batch;
  }

  template <typename F, typename Async, typename... Args>
  auto operator()(F f, Async&& async, Args&&... args) {
    return (*this)(f, async, args...);
  }

 private:
  std::size_t batch_size = 1;
  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
};

} /* namespace asio */
#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
} /* namespace cuda */
#endif  // KUL_GPU_FN_PER_NS
}  // namespace kul::gpu

#endif /* _KUL_GPU_CUDA_ASIO_HPP_ */
