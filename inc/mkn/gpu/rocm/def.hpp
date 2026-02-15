#ifndef _MKN_GPU_ROCM_DEF_HPP_
#define _MKN_GPU_ROCM_DEF_HPP_

#include "mkn/kul/log.hpp"
#include "mkn/gpu/def.hpp"

#include "hip/hip_runtime.h"

#include <string>

#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
#define MKN_GPU_NS mkn::gpu::hip
#else
#define MKN_GPU_NS mkn::gpu
#endif  // MKN_GPU_FN_PER_NS

namespace MKN_GPU_NS {

static_assert(CompileFlags::withROCM);

#define MKN_GPU_ASSERT(ans)               \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(hipError_t code, char const* file, int line, bool abort = true) {
  if (code != hipSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
    if (abort) std::abort();
  }
}

std::string getErrorString(auto const code) { return hipGetErrorString(code); }

std::uint32_t inline getWarpSize(size_t dev = 0) {
#ifdef _MKN_GPU_WARP_SIZE_
  return _MKN_GPU_WARP_SIZE_;
#else
  hipDeviceProp_t devProp;
  [[maybe_unused]] auto ret = hipGetDeviceProperties(&devProp, dev);
  return devProp.warpSize;
#endif /*_MKN_GPU_WARP_SIZE_    */
}

static std::uint32_t inline const warp_size = getWarpSize();

void inline setLimitMallocHeapSize(std::size_t const& bytes) {
  MKN_GPU_ASSERT(hipDeviceSetLimit(hipLimitMallocHeapSize, bytes));
}

void inline setDevice(std::size_t const& dev) { MKN_GPU_ASSERT(hipSetDevice(dev)); }

void inline sync() { MKN_GPU_ASSERT(hipDeviceSynchronize()); }
void inline sync(hipStream_t stream) { MKN_GPU_ASSERT(hipStreamSynchronize(stream)); }

template <typename Size>
void alloc(void*& p, Size size) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMalloc((void**)&p, size));
}

template <typename T, typename Size>
void alloc(T*& p, Size size) {
  KLOG(TRC) << "GPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(hipMalloc((void**)&p, size * sizeof(T)));
}

template <typename T, typename Size>
void alloc_host(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(hipHostMalloc((void**)&p, size * sizeof(T)));
}

template <typename T, typename Size>
void alloc_managed(T*& p, Size size) {
  auto const bytes = size * sizeof(T);
  KLOG(TRC) << "GPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(hipMallocManaged((void**)&p, bytes));
}

void inline destroy(void* p) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipFree(p));
}

template <typename T>
void destroy(T* ptr) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipFree(ptr));
}

template <typename T>
void destroy_host(T* ptr) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipHostFree(ptr));
}

template <typename T, typename Size>
void copy_on_device(T* dst, T const* src, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemcpy(dst, src, size * sizeof(T), hipMemcpyDeviceToDevice));
}

template <typename Size>
void send(void* p, void* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemcpy(p, t, size, hipMemcpyHostToDevice));
}

template <typename T, typename Size>
void send(T* p, T const* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemcpy(p, t, size * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T, typename Size>
void send_async(T* p, T const* t, hipStream_t& stream, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemcpyAsync(p, t, size * sizeof(T), hipMemcpyHostToDevice, stream));
}

template <typename T, typename Size>
void take(T const* p, T* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemcpy(t, p, size * sizeof(T), hipMemcpyDeviceToHost));
}

template <typename T, typename Size>
void take_async(T const* p, T* t, hipStream_t& stream, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemcpyAsync(t, p, size * sizeof(T), hipMemcpyDeviceToHost, stream));
}

}  // namespace MKN_GPU_NS

#endif /*_MKN_GPU_ROCM_DEF_HPP_*/
