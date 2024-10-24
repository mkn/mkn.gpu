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
#include "mkn/kul/assert.hpp"
#include "mkn/kul/threads.hpp"

#include "mkn/gpu/cli.hpp"
#include "mkn/gpu/def.hpp"

#include <cassert>
#include <cstring>

#define MKN_GPU_ASSERT(x) (KASSERT((x)))

#if defined(__device__)
#pragma message("__device__ already defined")
#error  // check your compiler
#endif

#if defined(__host__)
#pragma message("__host__ already defined")
#error  // check your compiler
#endif

#if defined(__global__)
#pragma message("__global__ already defined")
#error  // check your compiler
#endif

// we need to exclude these for CPU only operations
#define __device__
#define __host__
#define __global__

#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
#define MKN_GPU_NS mkn::gpu::cpu
#else
#define MKN_GPU_NS mkn::gpu
#endif  // MKN_GPU_FN_PER_NS

namespace MKN_GPU_NS {

struct dim3 {
  dim3() {}
  dim3(std::size_t x) : x{x} {}
  dim3(std::size_t x, std::size_t y) : x{x}, y{y} {}
  dim3(std::size_t x, std::size_t y, std::size_t z) : x{x}, y{y}, z{z} {}

  std::size_t x = 1, y = 1, z = 1;
};

void setLimitMallocHeapSize(std::size_t const& /*bytes*/) {} /*noop*/

void setDevice(std::size_t const& /*dev*/) {} /*noop*/

auto supportsCooperativeLaunch(int const /*dev*/ = 0) {
  int supportsCoopLaunch = 0;
  return supportsCoopLaunch;
}

struct Stream {
  Stream() {}
  ~Stream() {}

  auto& operator()() { return stream; };
  void sync() {}

  std::size_t stream = 0;
};

struct StreamEvent {
  StreamEvent(Stream&) {}
  ~StreamEvent() {}

  auto& operator()(std::function<void()> fn = {}) {
    fn();
    return *this;
  }

  bool finished() const { return fin; }

  Stream stream;
  bool fin = 1;
  std::function<void()> _fn;
};

template <typename T>
struct Pointer {
  Pointer(T* _t) : t{_t} {}

  bool is_unregistered_ptr() const { return t == nullptr; }
  bool is_host_ptr() const { return true; }
  bool is_device_ptr() const { return false; }
  bool is_managed_ptr() const { return false; }

  T* t;
};

template <typename Size>
void alloc(void*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size;
  MKN_GPU_ASSERT(p = std::malloc(size));
}

template <typename T, typename Size>
void alloc(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(p = reinterpret_cast<T*>(std::malloc(size * sizeof(T))));
}

template <typename T, typename Size>
void alloc_host(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(p = reinterpret_cast<T*>(std::malloc(size * sizeof(T))));
}

template <typename T, typename Size>
void alloc_managed(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(p = reinterpret_cast<T*>(std::malloc(size * sizeof(T))));
}

void inline destroy(void* p) {
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

template <typename T, typename Size>
void copy_on_device(T* dst, T const* src, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(std::memcpy(dst, src, size * sizeof(T)));
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

void inline sync() {}

#include "mkn/gpu/alloc.hpp"
#include "mkn/gpu/device.hpp"

namespace detail {
static thread_local std::size_t idx = 0;
}

template <bool _sync = true, bool _coop = false, typename F, typename... Args>
void launch(F f, dim3 g, dim3 b, std::size_t /*ds*/, std::size_t /*stream*/, Args&&... args) {
  std::size_t N = (g.x * g.y * g.z) * (b.x * b.y * b.z);
  KLOG(TRC) << N;
  std::apply(
      [&](auto&&... params) {
        for (std::size_t i = 0; i < N; ++i) {
          f(params...);
          detail::idx++;
        }
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

struct GLauncher : public Launcher {
  GLauncher(std::size_t s, [[maybe_unused]] size_t dev = 0) : Launcher{dim3{}, dim3{}}, count{s} {
    b.x = 1024;
    g.x = s / b.x;
    if ((s % b.x) > 0) ++g.x;
  }

  std::size_t count;
};

void prinfo(std::size_t /*dev*/ = 0) { KOUT(NON) << "Psuedo GPU in use"; }

}  // namespace MKN_GPU_NS

namespace mkn::gpu::cpu {

template <typename SIZE = std::uint32_t /*max 4294967296*/>
SIZE idx() {
  return MKN_GPU_NS::detail::idx;
}

}  // namespace mkn::gpu::cpu

namespace MKN_GPU_NS {

template <typename F, typename... Args>
static void global_gd_kernel(F& f, std::size_t s, Args... args) {
  if (auto i = mkn::gpu::cpu::idx(); i < s) f(args...);
}

#include "launchers.hpp"

void grid_sync() {}

} /* namespace MKN_GPU_NS */

#undef MKN_GPU_ASSERT
#endif /* _MKN_PSUEDO_GPU_HPP_ */
