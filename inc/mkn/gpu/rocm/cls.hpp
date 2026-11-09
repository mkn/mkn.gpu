// IWYU pragma: private, include "mkn/gpu.hpp"
// IWYU pragma: friend ".*mkn/gpu/(cpu|cuda|rocm|any)(/.*|\.hpp)"

#ifndef MKN_GPU_ROCM_CLS_HPP
#define MKN_GPU_ROCM_CLS_HPP

// device.ipp uses devmem_replace from itself, iwyu credits it to the public header
// IWYU pragma: no_include "mkn/gpu.hpp"

#include "mkn/kul/span.hpp"  // IWYU pragma: keep
#include "mkn/kul/tuple.hpp"
#include <vector>

#include "mkn/gpu/any/def.hpp"  // IWYU pragma: export
#include "mkn/gpu/any/cls.hpp"  // IWYU pragma: export
#include "hip/hip_runtime.h"
#include "mkn/gpu/rocm/def.hpp"  // IWYU pragma: export
#include "mkn/kul/log.hpp"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace MKN_GPU_NS {

struct Stream {
  Stream() { MKN_GPU_ASSERT(result = hipStreamCreate(&stream)); }
  ~Stream() { MKN_GPU_ASSERT(result = hipStreamDestroy(stream)); }

  auto& operator()() { return stream; };

  void sync() { MKN_GPU_ASSERT(result = hipStreamSynchronize(stream)); }

  hipError_t result;
  hipStream_t stream;
};

//

struct StreamEvent {
  //
  StreamEvent(Stream& stream_) : stream{stream_} {}
  StreamEvent(StreamEvent&& that) = default;
  StreamEvent(StreamEvent const&) = delete;
  StreamEvent& operator=(StreamEvent const&) = delete;

  auto& operator()(std::function<void()> fn = {}) {
    fin = 0;
    _fn = fn;
    MKN_GPU_ASSERT(hipStreamAddCallback(stream(), StreamEvent::Callback, this, 0));
    return *this;
  }

  static void Callback(hipStream_t /*stream*/, hipError_t /*status*/, void* ptr) {
    auto& self = *reinterpret_cast<StreamEvent*>(ptr);
    self._fn();
    self._fn = [] {};
    self.fin = 1;
  }

  bool finished() const { return fin; }

  Stream& stream;
  hipError_t result;
  std::function<void()> _fn;
  bool fin = 0;
};

// work queued while capturing is recorded rather than run, launch then replays all of it with
//  one call, saving the per launch CPU overhead. Kernel arguments are fixed at capture, pointers
//  can be captured but host values read at replay need a new capture. Capture is thread local,
//  other threads keep using their own streams normally meanwhile. See doc/tracing/graphs.md
struct Graph {
  Graph() = default;
  Graph(Graph const&) = delete;
  Graph& operator=(Graph const&) = delete;
  ~Graph() {
    reset();
    for (auto& e : events) MKN_GPU_ASSERT(hipEventDestroy(e));
  }

  template <typename Fn>
  Graph& capture(Stream& s, Fn&& fn) {
    begin(s);
    fn();
    return end(s);
  }

  // work on any of the streams, they become parallel branches of one graph. The other streams
  //  join the capture by waiting on an event of the first, and rejoin it before the end
  template <typename Fn>
  Graph& capture(std::vector<Stream>& streams, Fn&& fn) {
    if (streams.empty()) return *this;
    auto& origin = streams[0];
    while (events.size() < streams.size()) {
      hipEvent_t e;
      MKN_GPU_ASSERT(hipEventCreateWithFlags(&e, hipEventDisableTiming));
      events.emplace_back(e);
    }
    begin(origin);
    MKN_GPU_ASSERT(hipEventRecord(events[0], origin()));
    for (std::size_t i = 1; i < streams.size(); ++i)
      MKN_GPU_ASSERT(hipStreamWaitEvent(streams[i](), events[0], 0));
    fn();
    for (std::size_t i = 1; i < streams.size(); ++i) {
      MKN_GPU_ASSERT(hipEventRecord(events[i], streams[i]()));
      MKN_GPU_ASSERT(hipStreamWaitEvent(origin(), events[i], 0));
    }
    return end(origin);
  }

  // asynchronous, sync the stream to wait, for a multi stream capture the first
  Graph& launch(Stream& s) {
    MKN_GPU_ASSERT(hipGraphLaunch(exec, s()));
    return *this;
  }

  std::size_t nodes() const {
    std::size_t n = 0;
    if (graph) MKN_GPU_ASSERT(hipGraphGetNodes(graph, nullptr, &n));
    return n;
  }

  bool empty() const { return exec == nullptr; }

  void reset() {
    if (exec) MKN_GPU_ASSERT(hipGraphExecDestroy(exec));
    if (graph) MKN_GPU_ASSERT(hipGraphDestroy(graph));
    exec = nullptr;
    graph = nullptr;
  }

  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  std::vector<hipEvent_t> events;  // multi stream fork and join

 private:
  void begin(Stream& s) {
    reset();
    MKN_GPU_ASSERT(hipStreamBeginCapture(s(), hipStreamCaptureModeThreadLocal));
  }

  Graph& end(Stream& s) {
    MKN_GPU_ASSERT(hipStreamEndCapture(s(), &graph));
    MKN_GPU_ASSERT(hipGraphInstantiateWithFlags(&exec, graph, 0));
    return *this;
  }
};

// fn(data) on the host once prior work on the stream completes, fn must not call the GPU API
void inline launch_host_fn(Stream& s, void (*fn)(void*), void* data) {
  MKN_GPU_ASSERT(hipLaunchHostFunc(s(), fn, data));
}

// https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html#gaea86e91d3cd65992d787b39b218435a3
template <typename T>
struct Pointer {
  Pointer(T* _t) : t{_t} {
    if (!t) throw std::runtime_error("invalid nullptr");
#if HIP_VERSION_MAJOR >= 6
    MKN_GPU_ASSERT(hipPointerGetAttributes(&attributes, t));
#else
    // unregistered host memory is an error before ROCm 6, type stays hipMemoryTypeHost (0)
    if (auto const ret = hipPointerGetAttributes(&attributes, t); ret == hipErrorInvalidValue)
      (void)hipGetLastError();  // clear
    else
      MKN_GPU_ASSERT(ret);
#endif
  }
  bool is_host_ptr() const {
#if HIP_VERSION_MAJOR >= 6
    return type() == hipMemoryTypeUnregistered || type() == hipMemoryTypeHost;
#else
    return type() == hipMemoryTypeHost;
#endif
  }
  bool is_device_ptr() const {
    return type() == hipMemoryTypeDevice || type() == hipMemoryTypeArray;
  }
  bool is_managed_ptr() const {
    return type() == hipMemoryTypeManaged || type() == hipMemoryTypeUnified;
  }
  auto type() const { return attributes.type; }

  T* t;
  hipPointerAttribute_t attributes{};
};

#include "mkn/gpu/any/inc/alloc.ipp"   // IWYU pragma: export
#include "mkn/gpu/any/inc/device.ipp"  // IWYU pragma: export

template <bool _sync = true, typename F, typename... Args>
void launch(F&& f, dim3 g, dim3 b, std::size_t ds, hipStream_t& s, Args&&... args) {
  std::size_t N = (g.x * g.y * g.z) * (b.x * b.y * b.z);
  KLOG(TRC) << N;
  std::apply(
      [&](auto&&... params) {
        hipLaunchKernelGGL(f, g, b, ds, s, params...);
        MKN_GPU_ASSERT(hipGetLastError());
      },
      devmem_replace(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()));
  if constexpr (_sync) {
    if (s)
      sync(s);
    else
      sync();
  }
}

// https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/HIP-GUIDE.html#calling-global-functions
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
  hipStream_t s = 0;
};

}  // namespace MKN_GPU_NS

#endif /*MKN_GPU_ROCM_CLS_HPP*/
