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
#ifndef _MKN_GPU_LAUNCHERS_HPP_
#define _MKN_GPU_LAUNCHERS_HPP_

struct GDLauncher : public GLauncher {
  GDLauncher(std::size_t s, size_t dev = 0) : GLauncher{s, dev} {}

  template <typename F, typename... Args>
  auto operator()(F&& f, Args&&... args) {
    _launch(std::forward<F>(f),
            as_values(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()),
            count, args...);
  }

 protected:
  template <std::size_t... I, typename... Args>
  auto as_values(std::tuple<Args&...>&& tup, std::index_sequence<I...>) {
    using T = std::tuple<decltype(MKN_GPU_NS::replace(std::get<I>(tup)))&...>*;
    return T{nullptr};
  }

  template <typename F, typename... PArgs, typename... Args>
  void _launch(F&& f, std::tuple<PArgs&...>*, Args&&... args) {
    MKN_GPU_NS::launch(&global_gd_kernel<F, PArgs...>, g, b, ds, s, f, args...);
  }
};

#endif /* _MKN_GPU_LAUNCHERS_HPP_ */
