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
#ifndef _KUL_GPU_ROCM_ASIO_HPP_
#define _KUL_GPU_ROCM_ASIO_HPP_

#include "kul/gpu/tuple.hpp"

#define KUL_GPU_ASSERT(x) (KASSERT((x) == hipSuccess))

// review
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu

namespace kul::gpu {
#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
namespace hip {
#endif  // KUL_GPU_FN_PER_NS

namespace asio {

template <typename SIZE = uint32_t /*max 4294967296*/>
__device__ SIZE idx() {
  return hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
}

template <typename Batch>
struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(size_t tpx) : Launcher{dim3(), dim3(tpx)} {}

  template <typename F>
  auto& operator()(F f, Batch& batch) {
    auto& _asio = batch._asio;
    auto& streams = batch.streams;
    auto& streamSize = batch.streamSize;

    for (std::size_t i = 0; i < streams.size(); ++i) batch(i);

    auto _g = g;
    assert(g.x == _g.x);
    _g.x = streamSize / b.x;

    // for (std::size_t i = 0; i < streams.size(); ++i)
    //   hipLaunchKernelGGL(f, /*g*/ _g, b, ds, streams[i](), batch(), (i * streamSize));

    // std::apply([&](auto&&... params) { hipLaunchKernelGGL(f, g, b, ds, s, params...); },
    //            devmem_replace(std::forward_as_tuple(args...),
    //                           std::make_index_sequence<sizeof...(Args)>()));

    return batch;
  }

  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
};

} /* namespace asio */
#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
} /* namespace hip */
#endif  // KUL_GPU_FN_PER_NS
}  // namespace kul::gpu

#endif /* _KUL_GPU_ROCM_ASIO_HPP_ */
