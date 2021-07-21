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
#ifndef _KUL_GPU_ASIO_HPP_
#define _KUL_GPU_ASIO_HPP_

#if defined(KUL_GPU_ROCM)
// #include "kul/gpu/rocm/asio.hpp"
#elif defined(KUL_GPU_CUDA)
// #include "kul/gpu/cuda/asio.hpp"
#elif !defined(KUL_GPU_FN_PER_NS) || KUL_GPU_FN_PER_NS == 0
#error "UNKNOWN GPU / define KUL_GPU_ROCM or KUL_GPU_CUDA"
#endif

#include "kul/gpu/tuple.hpp"

namespace kul::gpu::asio {

template <typename ASYNC_t, typename SyncTuple>
struct Batch {
  using async_value_type = typename ASYNC_t::value_type;
  using SIZE = std::uint32_t;

  void buffered_alloc_size() {}

  Batch(std::size_t _batch_size, std::size_t _extra, ASYNC_t& async, SyncTuple&& rest)
      : batch_size{_batch_size},
        streamSize{async.size() / _batch_size},
        pinned{async},
        sync_{rest},
        streams{batch_size},
        _asio{pinned.size() + _extra} {
    static_assert(kul::is_span_like_v<ASYNC_t>);
    assert(pinned.size() % streams.size() == 0);  // NOPE
  }

  auto& operator[](std::size_t i) {
    assert(i < streams.size());
    streams[i].sync();
    return spans[i];
  }

  auto& get(std::size_t i) { return (*this)[i]; }

  void async_back() {
    clear();
    _async_back = std::make_unique<HostMem<async_value_type, SIZE>>(pinned.size());
    for (std::size_t i = 0; i < streams.size(); ++i) {
      auto offset = i * streamSize;
      auto& span = spans.emplace_back(_async_back->data() + offset, streamSize);
      _asio.take(streams[i], span, offset);
    }
  }

  void clear() {
    spans.clear();
    _async_back.release();
  }

  void operator()(std::size_t batch_idx) {
    _asio.send(streams[batch_idx], pinned.data(), streamSize, batch_idx * streamSize);
  }

  auto operator()() { return std::tuple_cat(std::make_tuple(_asio.p), sync_); }

  auto const& size() const { return streamSize; }

  std::size_t batch_size, streamSize;

  ASYNC_t const& pinned;

  std::vector<Stream> streams;
  AsioDeviceMem<async_value_type, SIZE> _asio;
  std::vector<kul::gpu::Span<async_value_type, SIZE>> spans;
  std::unique_ptr<HostMem<async_value_type, SIZE>> _async_back;

  SyncTuple sync_;
};

}  // namespace kul::gpu::asio

#if defined(KUL_GPU_ROCM)
#include "kul/gpu/rocm/asio.hpp"
#elif defined(KUL_GPU_CUDA)
#include "kul/gpu/cuda/asio.hpp"
#endif

#endif /* _KUL_GPU_ASIO_HPP_ */
