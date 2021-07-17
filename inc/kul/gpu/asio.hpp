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

template <typename ASYNC_t>
struct ChainLoader {
  using async_value_type = typename ASYNC_t::value_type;
  using SIZE = std::uint32_t;

  ChainLoader(std::size_t batches_, std::size_t items_) : batches{batches_}, items{items_} {}

  std::size_t batches = 0, items = 0;

  ASYNC_t* ptr = nullptr;

  std::function<void()> load{};
  std::vector<ASYNC_t const*> data_ptrs{};
  std::vector<std::size_t> offsets{};

  AsioDeviceMem<async_value_type, SIZE> _asio;
  std::vector<kul::gpu::Span<async_value_type, SIZE>> spans;

  auto operator()() { return ptr; }
  void operator()(async_value_type const* data, SIZE size) { spans.emplace_back(data, size); }
  void operator()(kul::gpu::Span<async_value_type, SIZE>&& data) { spans.emplace_back(data); }
};

// template <typename ASYNC_t, typename... SYNC_ts>
// struct BatchTask {
//   using async_value_type = typename ASYNC_t::value_type;
//   using sync_tuple = std::tuple<typename SYNC_ts::value_type...>;
//   using SIZE = std::uint32_t;

//   BatchTask(SIZE _batch_size, ASYNC_t const& _pinned, SYNC_ts const&... _inputs)
//       : batch_size{_batch_size},
//         pinned{_pinned},
//         inputs{std::forward_as_tuple(_inputs...)},
//         streams{batch_size} {
//     static_assert(kul::is_span_like_v<ASYNC_t> && (kul::is_span_like_v<SYNC_ts> && ...));
//     assert(pinned.size() % streams.size() == 0);  // NOPE
//     streamSize = pinned.size() / streams.size();
//   }

//   auto& operator[](std::size_t i) {
//     assert(i < streams.size());
//     streams[i].sync();
//     return back_spans[i];
//   }

//   void operator()(std::size_t batch_idx) {
//     assert(loader);
//     // loader(streams[i], pinned.data(), batch_idx);
//   }

//   auto operator()() { return *(loader)(); }

//   void async_back() {
//     clear();
//     _async_back = std::make_unique<HostMem<async_value_type, SIZE>>(pinned.size());
//     for (std::size_t i = 0; i < streams.size(); ++i) {
//       auto offset = i * streamSize;
//       auto& span = back_spans.emplace_back(_async_back->data() + offset, streamSize);
//       _asio.take(streams[i], span, offset);
//     }
//   }

//   void clear() {
//     back_spans.clear();
//     _async_back.release();
//   }

//   SIZE batch_size, streamSize;

//   ASYNC_t const& pinned;  // verify is_host_mem
//   std::tuple<SYNC_ts const&...> inputs{};

//   std::vector<Stream> streams;
//   AsioDeviceMem<async_value_type, SIZE> _asio;
//   std::vector<kul::gpu::Span<async_value_type, SIZE>> back_spans;
//   std::unique_ptr<HostMem<async_value_type, SIZE>> _async_back;

//   std::shared_ptr<ChainLoader<ASYNC_t>> loader = nullptr;
// };

// template <typename ASYNC_t, typename... SYNC_ts>
// struct Batch : BatchTask<ASYNC_t, SYNC_ts...> {
//   using SIZE = std::uint32_t;
//   using async_value_type = typename ASYNC_t::value_type;
//   using sync_tuple = std::tuple<typename SYNC_ts::value_type...>;
//   using Super = BatchTask<ASYNC_t, SYNC_ts...>;
//   using Super::_asio;
//   using Super::pinned;
//   using Super::streams;
//   using Super::streamSize;

//   Batch(SIZE _batch_size, ASYNC_t const& _pinned, SYNC_ts const&... _inputs)
//       : Super{_batch_size, _pinned, _inputs...}, _asio{_pinned.size()} {}

//   void operator()(std::size_t batch_idx) {
//     _asio.send(streams[batch_idx], pinned.data(), streamSize, batch_idx * streamSize);
//   }

//   auto operator()() { return _asio.p; }

//   // AsioDeviceMem<async_value_type, SIZE> _asio;
// };

template <typename ASYNC_t, typename SyncTuple>
struct Batch {
  using async_value_type = typename ASYNC_t::value_type;
  using SIZE = std::uint32_t;

  Batch(std::size_t _batch_size, ASYNC_t& async, SyncTuple&& rest)
      : batch_size{_batch_size},
        pinned{async},
        sync_{rest},
        streams{batch_size},
        _asio{pinned.size()} {
    static_assert(kul::is_span_like_v<ASYNC_t>);
    assert(pinned.size() % streams.size() == 0);  // NOPE
    streamSize = pinned.size() / streams.size();
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
