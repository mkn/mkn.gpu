// IWYU pragma: private, include "mkn/gpu.hpp"
// IWYU pragma: friend ".*mkn/gpu/(cpu|cuda|rocm|any)(/.*|\.hpp)"

#ifndef MKN_GPU_CPU_CLS_HPP
#define MKN_GPU_CPU_CLS_HPP

// device.ipp uses devmem_replace from itself, iwyu credits it to the public header
// IWYU pragma: no_include "mkn/gpu.hpp"

#include "mkn/gpu/any/cls.hpp"  // IWYU pragma: export

#include "mkn/gpu/cpu/def.hpp"  // IWYU pragma: export
#include "mkn/kul/span.hpp"     // IWYU pragma: keep
#include "mkn/kul/log.hpp"

#include <tuple>
#include <memory>
#include <utility>
#include <vector>
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

#include "mkn/gpu/any/inc/alloc.ipp"   // IWYU pragma: export
#include "mkn/gpu/any/inc/device.ipp"  // IWYU pragma: export

namespace detail {
static thread_local std::size_t idx = 0;
// set while this thread captures a Graph, launches are recorded rather than run
inline thread_local std::vector<std::function<void()>>* capture = nullptr;

// params are the kernel arguments after devmem_replace, by value as a GPU launch passes them
template <typename F, typename Params>
void run_kernel(F& f, dim3 const g, dim3 const b, Params& params) {
  gridDim = g;
  blockDim = b;
  std::apply(
      [&](auto&&... ps) {
        for (blockIdx.z = 0; blockIdx.z < g.z; ++blockIdx.z)
          for (blockIdx.y = 0; blockIdx.y < g.y; ++blockIdx.y)
            for (blockIdx.x = 0; blockIdx.x < g.x; ++blockIdx.x)
              for (threadIdx.z = 0; threadIdx.z < b.z; ++threadIdx.z)
                for (threadIdx.y = 0; threadIdx.y < b.y; ++threadIdx.y)
                  for (threadIdx.x = 0; threadIdx.x < b.x; ++threadIdx.x) {
                    f(ps...);
                    ++idx;
                  }
      },
      params);

  idx = 0;
  blockIdx = threadIdx = dim3{0, 0, 0};
}
}  // namespace detail

template <bool _sync = true, typename F, typename... Args>
void launch(F f, dim3 g, dim3 b, std::size_t /*ds*/, std::size_t /*stream*/, Args&&... args) {
  KLOG(TRC) << (g.x * g.y * g.z) * (b.x * b.y * b.z);
  auto params =
      devmem_replace(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>());
  if (detail::capture)  // recorded, see Graph
    detail::capture->emplace_back([=]() mutable { detail::run_kernel(f, g, b, params); });
  else
    detail::run_kernel(f, g, b, params);
}

// as the GPU Graph, kernel launches between begin and end are recorded and run by launch.
//  Only kernel launches are recorded, copies or a Graph launch during capture run immediately
struct Graph {
  Graph() = default;
  Graph(Graph const&) = delete;
  Graph& operator=(Graph const&) = delete;

  Graph& begin(Stream&) {
    reset();
    detail::capture = &fns;
    return *this;
  }

  Graph& end(Stream&) {
    detail::capture = nullptr;
    captured = true;
    return *this;
  }

  template <typename Fn>
  Graph& capture(Stream& s, Fn&& fn) {
    begin(s);
    fn();
    return end(s);
  }

  Graph& launch(Stream&) {
    for (auto& fn : fns) fn();
    return *this;
  }

  std::size_t nodes() const { return fns.size(); }
  bool empty() const { return !captured; }

  void reset() {
    fns.clear();
    captured = false;
  }

  std::vector<std::function<void()>> fns;
  bool captured = false;
};

// launch dimensions and native stream, see GlobalLauncher/DeviceLauncher
struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(std::size_t w, std::size_t h, std::size_t tpx, std::size_t tpy)
      : Launcher{dim3(mkn::gpu::grid_dim(w, tpx), mkn::gpu::grid_dim(h, tpy)), dim3(tpx, tpy)} {}
  Launcher(std::size_t x, std::size_t y, std::size_t z, std::size_t tpx, std::size_t tpy,
           std::size_t tpz)
      : Launcher{dim3(mkn::gpu::grid_dim(x, tpx), mkn::gpu::grid_dim(y, tpy),
                      mkn::gpu::grid_dim(z, tpz)),
                 dim3(tpx, tpy, tpz)} {}

  std::size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  std::size_t s = 0;
};

}  // namespace MKN_GPU_NS

#endif /*MKN_GPU_CPU_CLS_HPP*/
