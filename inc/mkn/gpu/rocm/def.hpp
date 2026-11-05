#ifndef _MKN_GPU_ROCM_DEF_HPP_
#define _MKN_GPU_ROCM_DEF_HPP_

#include "mkn/kul/log.hpp"
#include "mkn/gpu/def.hpp"

#include <string>
#include <cstdint>

#include "hip/hip_runtime.h"

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

auto inline getDeviceProperties(size_t dev = 0) {
  hipDeviceProp_t devProp;
  MKN_GPU_ASSERT(hipGetDeviceProperties(&devProp, dev));
  return devProp;
}

std::uint32_t inline getWarpSize(size_t dev = 0) {
#ifdef _MKN_GPU_WARP_SIZE_
  return _MKN_GPU_WARP_SIZE_;

#else
  hipDeviceProp_t devProp;
  MKN_GPU_ASSERT(hipGetDeviceProperties(&devProp, dev));
  return devProp.warpSize;

#endif /*_MKN_GPU_WARP_SIZE_    */
}

static std::uint32_t inline const warp_size = getWarpSize();

std::size_t inline getMaxThreadsPerBlock(std::size_t const dev = 0) {
  int threads = 0;
  MKN_GPU_ASSERT(hipDeviceGetAttribute(&threads, hipDeviceAttributeMaxThreadsPerBlock, dev));
  return threads;
}

// most threads per block a launch of kernel f can use on the current device, limited by its
//  register and shared memory use
std::size_t inline getKernelMaxThreadsPerBlock(void const* const f) {
  hipFuncAttributes attr{};
  MKN_GPU_ASSERT(hipFuncGetAttributes(&attr, f));
  return attr.maxThreadsPerBlock;
}

// dynamic shared memory per block a kernel gets without opting in
std::size_t inline getMaxSharedMemoryPerBlock(std::size_t const dev = 0) {
  int bytes = 0;
  MKN_GPU_ASSERT(hipDeviceGetAttribute(&bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, dev));
  return bytes;
}

// most dynamic shared memory per block a kernel can opt in to, see setKernelMaxDynamicSharedMemory
std::size_t inline getMaxSharedMemoryPerBlockOptin(std::size_t const dev = 0) {
  int bytes = 0;
  MKN_GPU_ASSERT(hipDeviceGetAttribute(&bytes, hipDeviceAttributeSharedMemPerBlockOptin, dev));
  return bytes;
}

// allows launches of kernel f with up to bytes of dynamic shared memory
void inline setKernelMaxDynamicSharedMemory(void const* const f, std::size_t const bytes) {
  MKN_GPU_ASSERT(hipFuncSetAttribute(f, hipFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(bytes)));
}

auto inline getLimitMallocHeapSize() {
  std::size_t bytes = 0;
  MKN_GPU_ASSERT(hipDeviceGetLimit(&bytes, hipLimitMallocHeapSize));
  return bytes;
}

void inline setLimitMallocHeapSize(std::size_t const& bytes) {
  MKN_GPU_ASSERT(hipDeviceSetLimit(hipLimitMallocHeapSize, bytes));
}

auto inline getLimitStackSize() {
  std::size_t bytes = 0;
  MKN_GPU_ASSERT(hipDeviceGetLimit(&bytes, hipLimitStackSize));
  return bytes;
}

void inline setLimitStackSize(std::size_t const& bytes) {
  MKN_GPU_ASSERT(hipDeviceSetLimit(hipLimitStackSize, bytes));
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

// migrate managed memory to device dev ahead of its use there, asynchronous on stream
template <typename T, typename Size>
void prefetch(T const* p, Size size, std::size_t const dev = 0, hipStream_t stream = 0) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemPrefetchAsync(p, size * sizeof(T), static_cast<int>(dev), stream));
}

// migrate managed memory to the host ahead of its use there, asynchronous on stream
template <typename T, typename Size>
void prefetch_host(T const* p, Size size, hipStream_t stream = 0) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemPrefetchAsync(p, size * sizeof(T), hipCpuDeviceId, stream));
}

}  // namespace MKN_GPU_NS

#endif /*_MKN_GPU_ROCM_DEF_HPP_*/
