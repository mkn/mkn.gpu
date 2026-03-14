#ifndef _MKN_GPU_CPU_CLS_HPP_
#define _MKN_GPU_CPU_CLS_HPP_

#include "mkn/gpu/cpu/def.hpp"

#include <tuple>
#include <memory>
#include <utility>
#include <functional>
#include <type_traits>

namespace MKN_GPU_NS {

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

#include "mkn/gpu/any/inc/alloc.ipp"
#include "mkn/gpu/any/inc/device.ipp"

namespace detail {
static thread_local std::size_t idx = 0;
}

template <bool _sync = true, typename F, typename... Args>
void launch(F f, dim3 g, dim3 b, std::size_t /*ds*/, std::size_t /*stream*/, Args&&... args) {
  std::size_t N = (g.x * g.y * g.z) * (b.x * b.y * b.z);
  KLOG(TRC) << N;
  std::apply(
      [&](auto&&... params) {
        for (std::size_t i = 0; i < N; ++i) {
          f(params...);
          ++blockIdx.x;
          ++detail::idx;
        }
      },
      devmem_replace(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()));

  detail::idx = 0;
  blockIdx.x = 0;
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
  GLauncher(std::size_t const s, std::size_t const /*dev*/ = 0)
      : Launcher{dim3{}, dim3{}}, count{s} {
    b.x = s;
  }

  std::size_t count;
};

}  // namespace MKN_GPU_NS

#endif /*_MKN_GPU_CPU_CLS_HPP_*/
