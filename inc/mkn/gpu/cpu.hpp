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
// IWYU pragma: private, include "mkn/gpu.hpp"
#ifndef _MKN_PSUEDO_GPU_HPP_
#define _MKN_PSUEDO_GPU_HPP_

#include "mkn/kul/log.hpp"
#include "mkn/kul/span.hpp"
#include "mkn/kul/tuple.hpp"
#include "mkn/kul/assert.hpp"

#include "mkn/gpu/def.hpp"

#include <cstring>

#define MKN_GPU_ASSERT(x) (KASSERT((x)))

#ifndef __device__
#define __device__
#endif  //__device__
#ifndef __host__
#define __host__
#endif  // host
#ifndef __global__
#define __global__
#endif  // __global__

namespace mkn::gpu {
#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
namespace cpu {
#define MKN_GPU_NS mkn::gpu::cpu
#else
#define MKN_GPU_NS mkn::gpu
#endif  // MKN_GPU_FN_PER_NS

struct dim3 {
  dim3() {}
  dim3(std::size_t x) : x{x} {}
  dim3(std::size_t x, std::size_t y) : x{x}, y{y} {}
  dim3(std::size_t x, std::size_t y, std::size_t z) : x{x}, y{y}, z{z} {}

  std::size_t x = 1, y = 1, z = 1;
};

struct Stream {
  Stream() {}
  ~Stream() {}

  auto& operator()() { return stream; };

  void sync() {}

  std::size_t stream = 0;
};

template <typename Size>
void alloc(void*& p, Size size) {
  KLOG(TRC) << "GPU alloced: " << size;
  MKN_GPU_ASSERT(p = std::malloc(size));
}

template <typename T, typename Size>
void alloc(T*& p, Size size) {
  KLOG(TRC) << "GPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(p = reinterpret_cast<T*>(std::malloc(size * sizeof(T))));
}

template <typename T, typename Size>
void alloc_host(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(p = reinterpret_cast<T*>(std::malloc(size * sizeof(T))));
}

void destroy(void* p) {
  KLOG(TRC);
  std::free(p);
}

template <typename T>
void destroy(T*& p) {
  KLOG(TRC);
  std::free(p);
}

template <typename T>
void destroy_host(T*& p) {
  KLOG(TRC);
  std::free(p);
}

template <typename Size>
void send(void* p, void* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(std::memcpy(p, t, size));
}

template <typename T, typename Size>
void send(T* p, T const* t, Size size = 1, Size start = 0) {
  KLOG(TRC);
  MKN_GPU_ASSERT(std::memcpy(p + start, t, size * sizeof(T)));
}

template <typename T, typename Size>
void take(T* p, T* t, Size size = 1, Size start = 0) {
  KLOG(TRC);
  MKN_GPU_ASSERT(std::memcpy(t, p + start, size * sizeof(T)));
}

template <typename T, typename Size>
void send_async(T* p, T const* t, Stream& /*stream*/, Size size = 1, Size start = 0) {
  KLOG(TRC);
  send(p, t, size, start);
}

template <typename T, typename Span>
void take_async(T* p, Span& span, Stream& /*stream*/, std::size_t start) {
  static_assert(mkn::kul::is_span_like_v<Span>);
  KLOG(TRC);
  take(p, span.data(), span.size(), start);
}

void sync() {}

#include "mkn/gpu/device.hpp"

namespace detail {
static std::size_t idx = 0;
}

template <typename F, typename... Args>
void launch(F&& f, dim3 g, dim3 b, std::size_t ds, std::size_t stream, Args&&... args) {
  std::size_t N = (g.x * g.y * g.z) * (b.x * b.y * b.z);

  std::apply(
      [&](auto&&... params) {
        for (std::size_t i = 0; i < N; ++i) f(params...);
      },
      devmem_replace(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()));

  detail::idx = 0;
}

struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(std::size_t w, std::size_t h, std::size_t tpx, std::size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}
  Launcher(std::size_t x, std::size_t y, std::size_t z, std::size_t tpx, std::size_t tpy,
           std::size_t tpz)
      : Launcher{dim3(x / tpx, y / tpy, z / tpz), dim3(tpx, tpy, tpz)} {}

  template <typename F, typename... Args>
  void operator()(F&& f, Args&&... args) {
    launch(f, g, b, ds, s, args...);
  }

  std::size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  std::size_t s = 0;
};

void prinfo(std::size_t dev = 0) { KOUT(NON) << "Psuedo GPU in use"; }

#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
} /* namespace cuda */
#endif  // MKN_GPU_FN_PER_NS
} /* namespace mkn::gpu */

namespace mkn::gpu::cpu {

template <typename SIZE = std::uint32_t /*max 4294967296*/>
__device__ SIZE idx() {
  return MKN_GPU_NS::detail::idx++;
}

}  // namespace mkn::gpu::cpu

#undef MKN_GPU_ASSERT
#endif /* _MKN_PSUEDO_GPU_HPP_ */