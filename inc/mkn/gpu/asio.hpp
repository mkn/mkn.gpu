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
#ifndef _MKN_GPU_ASIO_HPP_
#define _MKN_GPU_ASIO_HPP_

#if defined(MKN_GPU_ROCM)
#elif defined(MKN_GPU_CUDA)
#elif !defined(MKN_GPU_FN_PER_NS) || MKN_GPU_FN_PER_NS == 0
#error "UNKNOWN GPU / define MKN_GPU_ROCM or MKN_GPU_CUDA"
#endif

#include "mkn/gpu/tuple.hpp"

namespace mkn::gpu::asio {

struct BatchTuple {
  template <typename... Args>
  constexpr static auto type(Args&&... args) {
    auto tuple = std::forward_as_tuple(args...);
    auto constexpr tuple_size = std::tuple_size_v<decltype(tuple)>;
    return handle_inputs(tuple, std::make_index_sequence<tuple_size>());
  }
};

template <typename ASYNC_t, typename SyncTuple>
struct Batch {
  using async_value_type = typename std::decay_t<ASYNC_t>::value_type;
  using This = Batch<ASYNC_t, SyncTuple>;
  using SIZE = std::uint32_t;

  template <typename... Args>
  Batch(std::size_t _n_batches, ASYNC_t& async, Args&&... args)
      : n_batches{_n_batches},
        streamSize{async.size() / _n_batches},
        pinned{async},
        streams{n_batches},
        _asio{pinned.size()},
        sync_{BatchTuple::type(args...)} {
    static_assert(mkn::kul::is_span_like_v<ASYNC_t>);
    assert(pinned.size() > 0);
    assert(_asio.s > 0);
    assert(_asio.p != nullptr);
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
    if (!_async_back) _async_back = std::make_unique<HostMem<async_value_type>>(pinned.size());
    for (std::size_t i = 0; i < streams.size(); ++i) {
      auto offset = i * streamSize;
      auto& span = spans.emplace_back(_async_back->data() + offset, streamSize);
      _asio.take(streams[i], span, offset);
    }
  }

  void clear() { spans.clear(); }

  void operator()(std::size_t batch_idx) {
    _asio.send(streams[batch_idx], pinned.data(), streamSize, batch_idx * streamSize);
  }

  auto operator()() { return std::tuple_cat(std::forward_as_tuple(_asio), sync_); }

  void send() {
    for (std::size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) (*this)(batch_idx);
  }

  void sync() {
    for (auto& stream : streams) stream.sync();
  }

  auto const& size() const { return streamSize; }

  std::size_t n_batches, streamSize;

  ASYNC_t& pinned;

  std::vector<Stream> streams;
  AsioDeviceMem<async_value_type> _asio;
  std::vector<mkn::gpu::Span<async_value_type>> spans;
  std::unique_ptr<HostMem<async_value_type>> _async_back;

  SyncTuple sync_;
};

struct BatchMaker {
  template <typename ASYNC_t, typename... Args>
  static auto make_unique(std::size_t n, ASYNC_t& async, Args&&... args) {
    return std::make_unique<
        mkn::gpu::asio::Batch<ASYNC_t, decltype(mkn::gpu::asio::BatchTuple::type(args...))>>(
        n, async, args...);
  }
  template <typename ASYNC_t, typename... Args>
  static auto make_unique(ASYNC_t& async, Args&&... args) {
    return make_unique(1, async, args...);
  }
};

class Launcher {
  Launcher(dim3 _g, dim3 _b, std::size_t n_batches_) : n_batches{n_batches_}, g{_g}, b{_b} {}

 public:
  Launcher(std::size_t tpx, std::size_t n_batches_ = 1) : Launcher{dim3(), dim3(tpx), n_batches_} {}

  template <typename F, typename Batch_t>
  void launch(F&& f, Batch_t& batch) {
    auto const& streamSize = batch.streamSize;

    g.x = streamSize / b.x;
    if ((streamSize % b.x) > 0) ++g.x;

    for (std::size_t i = 0; i < n_batches; ++i) {
      if (m_send) batch(i);
      std::apply([&](auto&&... params) { launch(batch.streams[i], f, streamSize, i, params...); },
                 batch());
    }
    if (m_receive) batch.async_back();
  }

  auto& receive(bool b) {
    m_receive = b;
    return *this;
  }
  auto& send(bool b) {
    m_send = b;
    return *this;
  }

  template <typename F, typename Async, typename... Args>
  auto operator()(F&& f, Async& async, Args&&... args) {
    using Rest = decltype(BatchTuple::type(args...));
    auto batch_ptr = std::make_unique<Batch<Async, Rest>>(n_batches, async, args...);
    this->launch(f, *batch_ptr);
    return batch_ptr;
  }

 protected:
  template <typename F, typename... Args>
  __global__ static void kernel(F&& f, int max, int batch_index, Args... args) {
    auto bi = mkn::gpu::idx();
    if (bi < max) {
      f(bi + (max * batch_index), args...);
    }
  }

  template <std::size_t... I, typename... Args>
  auto as_values(std::tuple<Args&...>&& tup, std::index_sequence<I...>) {
    return (std::tuple<decltype(MKN_GPU_NS::replace(std::get<I>(tup)))&...>*){nullptr};
  }

  template <typename F, typename... PArgs, typename... Args>
  void _launch(F&& f, std::tuple<PArgs&...>*, Stream& stream, int max, int offset, Args&&... args) {
    MKN_GPU_NS::launch(kernel<F&&, PArgs...>, g, b, ds, stream(), f, max, offset, args...);
  }

  template <typename F, typename... Args>
  void launch(Stream& stream, F&& f, int max, int offset, Args&&... args) {
    _launch(f,
            as_values(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()),
            stream, max, offset, args...);
  }

 protected:
  std::size_t n_batches = 1;
  std::size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;

  bool m_send = 1, m_receive = 1;
};

}  // namespace mkn::gpu::asio

namespace std {
template <std::size_t index, typename ASYNC_t, typename SyncTuple>
auto& get(mkn::gpu::asio::Batch<ASYNC_t, SyncTuple> const& batch) {
  return std::get<index>(batch.sync_);
}

template <std::size_t index, typename ASYNC_t, typename SyncTuple>
auto& get(mkn::gpu::asio::Batch<ASYNC_t, SyncTuple>& batch) {
  return std::get<index>(batch.sync_);
}
}  // namespace std

#endif /* _MKN_GPU_ASIO_HPP_ */
