#ifndef _MKN_GPU_CUDA_CLS_HPP_
#define _MKN_GPU_CUDA_CLS_HPP_

#include <cstddef>
#include <functional>

#include <cuda_runtime.h>

#include "mkn/kul/log.hpp"
#include "mkn/kul/span.hpp"
#include "mkn/kul/tuple.hpp"

#include "mkn/gpu/any/def.hpp"
#include "mkn/gpu/any/cls.hpp"

#include "mkn/gpu/cuda/def.hpp"

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

#include "mkn/gpu/any/inc/alloc.ipp"
#include "mkn/gpu/any/inc/device.ipp"

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

struct Launcher {
  Launcher(dim3 _g, dim3 _b) : g{_g}, b{_b} {}
  Launcher(size_t w, size_t h, size_t tpx, size_t tpy)
      : Launcher{dim3(w / tpx, h / tpy), dim3(tpx, tpy)} {}
  Launcher(size_t x, size_t y, size_t z, size_t tpx, size_t tpy, size_t tpz)
      : Launcher{dim3(x / tpx, y / tpy, z / tpz), dim3(tpx, tpy, tpz)} {}

  template <typename F, typename... Args>
  void operator()(F&& f, Args&&... args) {
    launch(std::forward<F>(f), g, b, ds, s, args...);
  }

  size_t ds = 0 /*dynamicShared*/;
  dim3 g /*gridDim*/, b /*blockDim*/;
  cudaStream_t s = 0;
};

struct GLauncher : public Launcher {
  GLauncher(std::size_t const s, std::size_t const _dev = 0)
      : Launcher{dim3{}, dim3{}}, dev{_dev}, count{s} {
    MKN_GPU_ASSERT(cudaGetDeviceProperties(&devProp, dev));

    resize(s);
  }

  void resize(std::size_t const s, std::size_t const bx = 0) {
    b.x = bx > 0 ? bx : cli.bx_threads();
    g.x = s / b.x;
    if ((s % b.x) > 0) ++g.x;
  }

  std::size_t dev = 0;
  std::size_t count = 0;
  cudaDeviceProp devProp;
  mkn::gpu::Cli<cudaDeviceProp> cli{devProp};
};

}  // namespace MKN_GPU_NS

#endif /*_MKN_GPU_CUDA_CLS_HPP_*/
