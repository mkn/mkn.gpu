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
// IWYU pragma: private, include "mkn/gpu.hpp"
#ifndef _MKN_PSUEDO_GPU_HPP_
#define _MKN_PSUEDO_GPU_HPP_

#include "mkn/kul/log.hpp"
#include "mkn/kul/span.hpp"
#include "mkn/kul/tuple.hpp"

#include "mkn/gpu/cpu/def.hpp"
#include "mkn/gpu/cpu/api.hpp"
#include "mkn/gpu/cpu/cls.hpp"

#include <cassert>
#include <cstring>
#include <algorithm>

namespace MKN_GPU_NS {

template <typename Container, typename T>
void fill(Container& c, size_t const size, T const val) {
  std::fill(c.begin(), c.begin() + size, val);
}

template <typename Container, typename T>
void fill(Container& c, T const val) {
  fill(c, c.size(), val);
}

template <typename T>
void fill_warp_size(T* const t, std::size_t const size, T const val) {
  std::fill(t, t + size, val);
}

void inline prinfo(std::size_t /*dev*/ = 0) { KOUT(NON) << "Pseudo GPU in use"; }

}  // namespace MKN_GPU_NS

namespace mkn::gpu::cpu {

template <typename SIZE = std::uint32_t /*max 4294967296*/>
SIZE inline idx() {
  return MKN_GPU_NS::detail::idx;
}

}  // namespace mkn::gpu::cpu

namespace MKN_GPU_NS {

template <typename F, typename... Args>
static void global_gd_kernel(F& f, std::size_t s, Args... args) {
  if (auto i = mkn::gpu::cpu::idx(); i < s) f(args...);
}

template <typename F, typename... Args>
static void global_d_kernel(F& f, Args... args) {
  f(args...);
}

#include "mkn/gpu/any/inc/launchers.ipp"

} /* namespace MKN_GPU_NS */

#undef MKN_GPU_ASSERT
#endif /* _MKN_PSUEDO_GPU_HPP_ */
