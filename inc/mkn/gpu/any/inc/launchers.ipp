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
#ifndef _MKN_GPU_LAUNCHERS_HPP_
#define _MKN_GPU_LAUNCHERS_HPP_

namespace detail {

template <std::size_t... I, typename... Args>
auto _as_values(std::tuple<Args&...>&& tup, std::index_sequence<I...>) {
  using T = std::tuple<decltype(MKN_GPU_NS::replace(std::get<I>(tup)))&...>*;
  return T{nullptr};
}

template <typename... Args>
auto as_values(Args&... args) {
  return _as_values(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>());
}

// cached per thread and device, attribute queries are not free
std::size_t inline max_threads_per_block(std::size_t const dev) {
  thread_local std::vector<std::size_t> cache;
  if (cache.size() <= dev) cache.resize(dev + 1, 0);
  if (cache[dev] == 0) cache[dev] = getMaxThreadsPerBlock(dev);
  return cache[dev];
}

template <typename T>
concept span_like = mkn::kul::is_span_like_v<std::decay_t<T>>;

}  // namespace detail

// grid/block derived from a problem size, device functions are bounds checked against it
struct Guess {};

// grid/block provided by the caller
struct Fixed {};

// launches are asynchronous, call sync() to wait for the stream (or device without a stream)
template <typename Mode>
struct ALauncher : public Launcher {
  static constexpr bool guess = std::is_same_v<Mode, Guess>;
  static_assert(guess || std::is_same_v<Mode, Fixed>, "Launcher mode must be Guess or Fixed");

  ALauncher(std::size_t const size, std::size_t const dev = 0)
    requires(guess)
      : Launcher{dim3{}, dim3{}}, count{size} {
    b.x = mkn::gpu::bx_threads(detail::max_threads_per_block(dev));
    g.x = size / b.x;
    if ((size % b.x) > 0) ++g.x;
  }

  template <detail::span_like C>
  ALauncher(C const& c, std::size_t const dev = 0)
    requires(guess)
      : ALauncher{static_cast<std::size_t>(c.size()), dev} {}

  ALauncher()
    requires(!guess)
      : Launcher{dim3{1}, dim3{warp_size}} {}

  ALauncher(dim3 const _g, dim3 const _b)
    requires(!guess)
      : Launcher{_g, _b} {}

  ALauncher(std::size_t const w, std::size_t const h, std::size_t const tpx, std::size_t const tpy)
    requires(!guess)
      : Launcher{w, h, tpx, tpy} {}

  ALauncher(std::size_t const x, std::size_t const y, std::size_t const z, std::size_t const tpx,
            std::size_t const tpy, std::size_t const tpz)
    requires(!guess)
      : Launcher{x, y, z, tpx, tpy, tpz} {}

  auto& sync() {
    if (s)
      MKN_GPU_NS::sync(s);
    else
      MKN_GPU_NS::sync();
    return *this;
  }

  bool empty() const {
    if constexpr (guess) return count == 0;
    return false;
  }

  std::size_t count = 0;  // Guess only
};

// for __global__ functions
template <typename Mode>
struct GlobalLauncher : public ALauncher<Mode> {
  using Super = ALauncher<Mode>;
  using Super::Super;

  template <typename F, typename... Args>
  auto& operator()(F&& f, Args&&... args) {
    if (this->empty()) return *this;
    MKN_GPU_NS::launch<false>(std::forward<F>(f), this->g, this->b, this->ds, this->s, args...);
    return *this;
  }

  template <typename F, typename... Args>
  auto& stream(Stream& _s, F&& f, Args&&... args) {
    this->s = _s();
    return (*this)(std::forward<F>(f), args...);
  }
};

// for __device__ functions/lambdas
template <typename Mode>
struct DeviceLauncher : public ALauncher<Mode> {
  using Super = ALauncher<Mode>;
  using Super::Super;

  template <typename F, typename... Args>
  auto& operator()(F&& f, Args&&... args) {
    if (this->empty()) return *this;
    _launch(f, detail::as_values(args...), args...);
    return *this;
  }

  template <typename F, typename... Args>
  auto& stream(Stream& _s, F&& f, Args&&... args) {
    this->s = _s();
    return (*this)(std::forward<F>(f), args...);
  }

 protected:
  template <typename F, typename... PArgs, typename... Args>
  void _launch(F& f, std::tuple<PArgs&...>*, Args&&... args) {
    if constexpr (Super::guess)
      MKN_GPU_NS::launch<false>(&global_gd_kernel<F, PArgs...>, this->g, this->b, this->ds, this->s,
                                f, this->count, args...);
    else
      MKN_GPU_NS::launch<false>(&global_d_kernel<F, PArgs...>, this->g, this->b, this->ds, this->s,
                                f, args...);
  }
};

GlobalLauncher(std::size_t) -> GlobalLauncher<Guess>;
GlobalLauncher(std::size_t, std::size_t) -> GlobalLauncher<Guess>;
template <detail::span_like C>
GlobalLauncher(C const&) -> GlobalLauncher<Guess>;
template <detail::span_like C>
GlobalLauncher(C const&, std::size_t) -> GlobalLauncher<Guess>;
GlobalLauncher() -> GlobalLauncher<Fixed>;
GlobalLauncher(dim3, dim3) -> GlobalLauncher<Fixed>;
GlobalLauncher(std::size_t, std::size_t, std::size_t, std::size_t) -> GlobalLauncher<Fixed>;
GlobalLauncher(std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t)
    -> GlobalLauncher<Fixed>;

DeviceLauncher(std::size_t) -> DeviceLauncher<Guess>;
DeviceLauncher(std::size_t, std::size_t) -> DeviceLauncher<Guess>;
template <detail::span_like C>
DeviceLauncher(C const&) -> DeviceLauncher<Guess>;
template <detail::span_like C>
DeviceLauncher(C const&, std::size_t) -> DeviceLauncher<Guess>;
DeviceLauncher() -> DeviceLauncher<Fixed>;
DeviceLauncher(dim3, dim3) -> DeviceLauncher<Fixed>;
DeviceLauncher(std::size_t, std::size_t, std::size_t, std::size_t) -> DeviceLauncher<Fixed>;
DeviceLauncher(std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t)
    -> DeviceLauncher<Fixed>;

template <typename Mode>
using GLauncher = GlobalLauncher<Mode>;

template <typename Mode>
using DLauncher = DeviceLauncher<Mode>;

#endif /* _MKN_GPU_LAUNCHERS_HPP_ */
