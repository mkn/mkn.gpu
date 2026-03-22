#ifndef _MKN_GPU_ROCM_CLS_HPP_
#define _MKN_GPU_ROCM_CLS_HPP_

#include "mkn/kul/span.hpp"
#include "mkn/kul/tuple.hpp"

#include "mkn/gpu/any/def.hpp"
#include "mkn/gpu/any/cls.hpp"

#include "hip/hip_runtime.h"

#include "mkn/gpu/rocm/def.hpp"

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

// https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html#gaea86e91d3cd65992d787b39b218435a3
template <typename T>
struct Pointer {
  Pointer(T* _t) : t{_t} {
    if (!t) throw std::runtime_error("invalid nullptr");
    MKN_GPU_ASSERT(hipPointerGetAttributes(&attributes, t));
  }
  bool is_host_ptr() const { return type() == hipMemoryTypeHost; }
  bool is_device_ptr() const {
    return type() == hipMemoryTypeDevice || type() == hipMemoryTypeArray;
  }
  bool is_managed_ptr() const {
    return type() == hipMemoryTypeManaged || type() == hipMemoryTypeUnified;
  }
  auto type() const { return attributes.type; }

  T* t;
  hipPointerAttribute_t attributes;
};

#include "mkn/gpu/any/inc/alloc.ipp"
#include "mkn/gpu/any/inc/device.ipp"

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
  hipStream_t s = 0;
};

struct GLauncher : public Launcher {
  GLauncher(std::size_t const s, std::size_t const _dev = 0)
      : Launcher{dim3{}, dim3{}}, dev{_dev}, count{s} {
    MKN_GPU_ASSERT(hipGetDeviceProperties(&devProp, dev));

    resize(s);
  }

  void resize(std::size_t const s, std::size_t const bx = 0) {
    b.x = bx > 0 ? bx : cli.bx_threads();
    g.x = s / b.x;
    if ((s % b.x) > 0) ++g.x;
  }

  std::size_t dev = 0;
  std::size_t count = 0;
  hipDeviceProp_t devProp;
  mkn::gpu::Cli<hipDeviceProp_t> cli{devProp};
};

}  // namespace MKN_GPU_NS

#endif /*_MKN_GPU_ROCM_CLS_HPP_*/
