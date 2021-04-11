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

#include "kul/gpu/tuple.hpp"

#define KUL_GPU_ASSERT(x) (KASSERT((x) == cudaSuccess))

// review https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu

namespace kul::gpu {
#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
namespace cuda {
#endif  // KUL_GPU_FN_PER_NS

struct Stream{
  Stream(){
    KUL_GPU_ASSERT(result = cudaStreamCreate(&stream));
  }
  ~Stream(){
    KUL_GPU_ASSERT(result = cudaStreamDestroy(stream));
  }

  auto operator()(){ return stream; };

  void sync() { result = cudaStreamSynchronize(stream); }

  cudaError_t result;
  cudaStream_t stream;
};

template <typename T, typename SIZE = uint32_t>
struct AsioDeviceMem {

  AsioDeviceMem(SIZE _s, SIZE _nStreams) : s{_s},  nStreams{_nStreams}{
    SIZE alloc_bytes = s * sizeof(T);
    KLOG(OTH) << "GPU alloced: " << alloc_bytes;
    if (s) KUL_GPU_ASSERT(cudaMalloc((void**)&p, alloc_bytes));
  }

  ~AsioDeviceMem() {
    if (p && s) KUL_GPU_ASSERT(cudaFree(p));
  }

  void send(Stream &stream, T const* const t, SIZE _size = 1, SIZE start = 0) {
    KUL_GPU_ASSERT(cudaMemcpyAsync(p + start, t + start, _size * sizeof(T), cudaMemcpyHostToDevice, stream()));
  }

  template <typename Span>
  void take(Stream &stream, Span& span, std::size_t start)  {
    static_assert(kul::is_span_like_v<Span>);
    KUL_GPU_ASSERT(cudaMemcpyAsync(span.data(), p + start, span.size() * sizeof(T), cudaMemcpyDeviceToHost, stream()));
  }

  auto& size() const { return s; }

  SIZE s = 0;
  SIZE nStreams = 0;
  T* p = nullptr;
};


namespace asio {


template <typename SIZE = uint32_t /*max 4294967296*/>
__device__ SIZE idx() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}


template <typename ASYNC_t, typename... SYNC_ts>
struct Batch {
  using async_value_type = typename ASYNC_t::value_type;
  using sync_tuple = std::tuple<typename SYNC_ts::value_type...>;
  using SIZE = std::uint32_t;

  Batch(SIZE _batch_size, ASYNC_t const& _pinned, SYNC_ts const&... _inputs)
    : batch_size{_batch_size}, pinned{_pinned}, inputs{std::forward_as_tuple(_inputs...)}, streams{batch_size}, _asio{pinned.size(), _batch_size} {
    static_assert(kul::is_span_like_v<ASYNC_t> && (kul::is_span_like_v<SYNC_ts> && ...));
    assert(pinned.size() % streams.size() == 0); // NOPE
    streamSize = pinned.size() / streams.size();
  }

  auto& operator[](std::size_t i){
    assert(i < streams.size());
    streams[i].sync();
    return spans[i];
  }

  void async_back(){
    clear();
    _async_back = std::make_unique<HostMem<async_value_type, SIZE>>(pinned.size());
    for(std::size_t i = 0; i < streams.size(); ++i){
      auto offset = i * streamSize;
      auto& span = spans.emplace_back(_async_back->data() + offset, streamSize);
      _asio.take(streams[i], span, offset);
    }
  }

  void clear(){
    spans.clear();
    _async_back.release();
  }

  SIZE batch_size, streamSize;

  ASYNC_t const& pinned; // verify is_host_mem
  std::tuple<SYNC_ts const&...> inputs{};
  sync_tuple types{};

  std::vector<Stream> streams;
  AsioDeviceMem<async_value_type, SIZE> _asio;
  std::vector<kul::gpu::Span<async_value_type, SIZE>> spans;
  std::unique_ptr<HostMem<async_value_type, SIZE>> _async_back;
};

struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}

  Launcher(size_t tpx)
      : Launcher{dim3(), dim3(tpx)} {}

  Launcher(size_t w, size_t h, size_t tpx, size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}
  Launcher(size_t x, size_t y, size_t z, size_t tpx, size_t tpy, size_t tpz)
      : Launcher{dim3(x / tpx, y / tpy, z / tpz), dim3(tpx, tpy, tpz)} {}

  template <typename F, typename ...BatchArgs>
  auto& operator()(F f, Batch<BatchArgs...> & batch) {
    using Batch_t = Batch<BatchArgs...>;

    auto& _asio  = batch._asio;
    auto& streams = batch.streams;
    auto& streamSize = batch.streamSize;

    for (std::size_t i = 0; i < streams.size(); ++i)
      _asio.send(streams[i], batch.pinned.data(), streamSize, i * streamSize);

    auto _g = g;
    assert(g.x == _g.x);
    _g.x = streamSize / b.x;

    for (std::size_t i = 0; i < streams.size(); ++i)
        f<<<_g, b, ds, streams[i]()>>>(_asio.p, (i * streamSize));

    // std::apply([&](auto&&... params) { cudaLaunchKernelGGL(f, g, b, ds, s, params...); },
    //            devmem_replace(std::forward_as_tuple(args...),
    //                           std::make_index_sequence<sizeof...(Args)>()));

    return batch;
  }

  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
};


} /* namespace asio */
#if defined(KUL_GPU_FN_PER_NS) && KUL_GPU_FN_PER_NS
} /* namespace cuda */
#endif  // KUL_GPU_FN_PER_NS
} /* namespace kul::gpu::asio */

#endif /* _KUL_GPU_CUDA_ASIO_HPP_ */
