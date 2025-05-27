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

}  // namespace detail

template <bool _sync = true>
struct GDLauncher : public GLauncher {
  GDLauncher(std::size_t const s, size_t const dev = 0) : GLauncher{s, dev} {}

  template <typename F, typename... Args>
  auto operator()(F&& f, Args&&... args) {
    _launch(s, f, detail::as_values(args...), count, args...);
  }

  template <typename F, typename... Args>
  auto stream(Stream& s, F&& f, Args&&... args) {
    _launch(s.stream, f, detail::as_values(args...), count, args...);
  }

 protected:
  template <typename S, typename F, typename... PArgs, typename... Args>
  void _launch(S& _s, F& f, std::tuple<PArgs&...>*, Args&&... args) {
    MKN_GPU_NS::launch<_sync>(&global_gd_kernel<F, PArgs...>, g, b, ds, _s, f, args...);
  }
};

template <bool _sync = true>
struct DLauncher : public Launcher {
  DLauncher() : Launcher{dim3{1}, dim3{warp_size}} {}
  DLauncher(size_t const /*dev*/) : Launcher{{}, {}} {}

  template <typename... Args>
  DLauncher(Args&&... args)
    requires(sizeof...(Args) > 0)
      : Launcher{args...} {}

  template <typename F, typename... Args>
  auto operator()(F&& f, Args&&... args) {
    _launch(s, f, detail::as_values(args...), args...);
  }

  template <typename F, typename... Args>
  auto stream(Stream& s, F&& f, Args&&... args) {
    _launch(s.stream, f, detail::as_values(args...), args...);
  }

 protected:
  template <typename S, typename F, typename... PArgs, typename... Args>
  void _launch(S& _s, F& f, std::tuple<PArgs&...>*, Args&&... args) {
    MKN_GPU_NS::launch<_sync>(&global_d_kernel<F, PArgs...>, g, b, ds, _s, f, args...);
  }

  //
};

#endif /* _MKN_GPU_LAUNCHERS_HPP_ */
