// IWYU pragma: private, include "mkn/gpu.hpp"
// IWYU pragma: friend ".*mkn/gpu/(cpu|cuda|rocm|any)(/.*|\.hpp)"

#ifndef MKN_GPU_CUDA_CLS_HPP
#define MKN_GPU_CUDA_CLS_HPP

// device.ipp uses devmem_replace from itself, iwyu credits it to the public header
// IWYU pragma: no_include "mkn/gpu.hpp"

#include "mkn/kul/log.hpp"
#include "mkn/kul/span.hpp"  // IWYU pragma: keep
#include "mkn/kul/tuple.hpp"
#include <vector>

#include "mkn/gpu/any/def.hpp"   // IWYU pragma: export
#include "mkn/gpu/any/cls.hpp"   // IWYU pragma: export
#include "mkn/gpu/cuda/def.hpp"  // IWYU pragma: export

#include <cuda_runtime.h>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace MKN_GPU_NS {

struct Stream {
  Stream() { MKN_GPU_ASSERT(result = cudaStreamCreate(&stream)); }
  ~Stream() { MKN_GPU_ASSERT(result = cudaStreamDestroy(stream)); }

  auto& operator()() { return stream; };

  void sync() { MKN_GPU_ASSERT(result = cudaStreamSynchronize(stream)); }

  cudaError_t result;
  cudaStream_t stream;
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
    MKN_GPU_ASSERT(cudaStreamAddCallback(stream(), StreamEvent::Callback, this, 0));
    return *this;
  }

  static void Callback(cudaStream_t /*stream*/, cudaError_t /*status*/, void* ptr) {
    auto& self = *reinterpret_cast<StreamEvent*>(ptr);
    self._fn();
    self._fn = [] {};
    self.fin = 1;
  }

  bool finished() const { return fin; }

  Stream& stream;
  cudaError_t result;
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
    for (auto& e : events) MKN_GPU_ASSERT(cudaEventDestroy(e));
  }

  template <typename Fn>
  Graph& capture(Stream& s, Fn&& fn) {
    begin(s);
    try {
      fn();
    } catch (...) {
      cancel(s);
      throw;
    }
    return end(s);
  }

  // work on any of the streams, they become parallel branches of one graph. The other streams
  //  join the capture by waiting on an event of the first, and rejoin it before the end
  template <typename Fn>
  Graph& capture(std::vector<Stream>& streams, Fn&& fn) {
    if (streams.empty()) return *this;
    auto& origin = streams[0];
    while (events.size() < streams.size()) {
      cudaEvent_t e;
      MKN_GPU_ASSERT(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      events.emplace_back(e);
    }
    begin(origin);
    MKN_GPU_ASSERT(cudaEventRecord(events[0], origin()));
    for (std::size_t i = 1; i < streams.size(); ++i)
      MKN_GPU_ASSERT(cudaStreamWaitEvent(streams[i](), events[0], 0));
    try {
      fn();
    } catch (...) {
      cancel(origin, &streams);
      throw;
    }
    for (std::size_t i = 1; i < streams.size(); ++i) {
      MKN_GPU_ASSERT(cudaEventRecord(events[i], streams[i]()));
      MKN_GPU_ASSERT(cudaStreamWaitEvent(origin(), events[i], 0));
    }
    return end(origin);
  }

  // asynchronous, sync the stream to wait, for a multi stream capture the first
  Graph& launch(Stream& s) {
    MKN_GPU_ASSERT(cudaGraphLaunch(exec, s()));
    return *this;
  }

  std::size_t nodes() const {
    std::size_t n = 0;
    if (graph) MKN_GPU_ASSERT(cudaGraphGetNodes(graph, nullptr, &n));
    return n;
  }

  bool empty() const { return exec == nullptr; }

  void reset() {
    if (exec) MKN_GPU_ASSERT(cudaGraphExecDestroy(exec));
    if (graph) MKN_GPU_ASSERT(cudaGraphDestroy(graph));
    exec = nullptr;
    graph = nullptr;
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  std::vector<cudaEvent_t> events;  // multi stream fork and join

 private:
  void begin(Stream& s) {
    reset();
    MKN_GPU_ASSERT(cudaStreamBeginCapture(s(), cudaStreamCaptureModeThreadLocal));
  }

  // fn threw: the other streams rejoin and the capture ends, so every stream can be used again,
  //  the partial graph is dropped. Errors are ignored, the capture may already be invalidated
  void cancel(Stream& origin, std::vector<Stream>* const joined = nullptr) noexcept {
    if (joined)
      for (std::size_t i = 1; i < joined->size(); ++i) {
        (void)cudaEventRecord(events[i], (*joined)[i]());
        (void)cudaStreamWaitEvent(origin(), events[i], 0);
      }
    cudaGraph_t partial = nullptr;
    (void)cudaStreamEndCapture(origin(), &partial);
    if (partial) (void)cudaGraphDestroy(partial);
    (void)cudaGetLastError();
  }

  Graph& end(Stream& s) {
    MKN_GPU_ASSERT(cudaStreamEndCapture(s(), &graph));
    MKN_GPU_ASSERT(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    return *this;
  }
};

// fn(data) on the host once prior work on the stream completes, fn must not call the GPU API
void inline launch_host_fn(Stream& s, void (*fn)(void*), void* data) {
  MKN_GPU_ASSERT(cudaLaunchHostFunc(s(), fn, data));
}

//

template <typename T>
struct Pointer {
  Pointer(T* _t) : t{_t} {
    if (!t) throw std::runtime_error("invalid nullptr");
    MKN_GPU_ASSERT(cudaPointerGetAttributes(&attributes, t));
  }
  bool is_host_ptr() const {
    return type() == cudaMemoryTypeUnregistered or type() == cudaMemoryTypeHost;
  }
  bool is_device_ptr() const { return type() == cudaMemoryTypeDevice; }
  bool is_managed_ptr() const { return type() == cudaMemoryTypeManaged; }
  auto type() const { return attributes.type; }

  T* t;
  cudaPointerAttributes attributes;
};

#include "mkn/gpu/any/inc/alloc.ipp"   // IWYU pragma: export
#include "mkn/gpu/any/inc/device.ipp"  // IWYU pragma: export

template <bool _sync = true, typename F, typename... Args>
void launch(F&& f, dim3 g, dim3 b, std::size_t ds, cudaStream_t& s, Args&&... args) {
  std::size_t N = (g.x * g.y * g.z) * (b.x * b.y * b.z);
  KLOG(TRC) << "N=" << N << " ds=" << ds;
  std::apply(
      [&](auto&&... params) {
        f<<<g, b, ds, s>>>(params...);
        MKN_GPU_ASSERT(cudaGetLastError());
      },
      devmem_replace(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>()));
  if constexpr (_sync) {
    if (s)
      sync(s);
    else
      sync();
  }
}

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
  cudaStream_t s = 0;
};

}  // namespace MKN_GPU_NS

#endif /*MKN_GPU_CUDA_CLS_HPP*/
