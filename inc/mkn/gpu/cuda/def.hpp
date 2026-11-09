// IWYU pragma: private, include "mkn/gpu.hpp"
// IWYU pragma: friend ".*mkn/gpu/(cpu|cuda|rocm|any)(/.*|\.hpp)"

#ifndef MKN_GPU_CUDA_DEF_HPP
#define MKN_GPU_CUDA_DEF_HPP

#include "mkn/kul/log.hpp"
#include "mkn/gpu/def.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <vector>

#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
#define MKN_GPU_NS mkn::gpu::cuda
#else
#define MKN_GPU_NS mkn::gpu
#endif  // MKN_GPU_FN_PER_NS

namespace MKN_GPU_NS {

static_assert(CompileFlags::withCUDA);

#define MKN_GPU_ASSERT(ans)               \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, char const* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) std::abort();
  }
}

std::string getErrorString(auto const code) { return cudaGetErrorString(code); }

auto inline getDeviceProperties(size_t dev = 0) {
  cudaDeviceProp devProp;
  MKN_GPU_ASSERT(cudaGetDeviceProperties(&devProp, dev));
  return devProp;
}

std::uint32_t inline getWarpSize(size_t dev = 0) {
#ifdef MKN_GPU_WARP_SIZE
  return MKN_GPU_WARP_SIZE;

#else
  return getDeviceProperties(dev).warpSize;

#endif /*MKN_GPU_WARP_SIZE    */
}

static std::uint32_t inline const warp_size = getWarpSize();

std::size_t inline getMaxThreadsPerBlock(std::size_t const dev = 0) {
  int threads = 0;
  MKN_GPU_ASSERT(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, dev));
  return threads;
}

// most threads per block a launch of kernel f can use on the current device, limited by its
//  register and shared memory use
std::size_t inline getKernelMaxThreadsPerBlock(void const* const f) {
  cudaFuncAttributes attr{};
  MKN_GPU_ASSERT(cudaFuncGetAttributes(&attr, f));
  return attr.maxThreadsPerBlock;
}

// dynamic shared memory per block a kernel gets without opting in
std::size_t inline getMaxSharedMemoryPerBlock(std::size_t const dev = 0) {
  int bytes = 0;
  MKN_GPU_ASSERT(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, dev));
  return bytes;
}

// most dynamic shared memory per block a kernel can opt in to, see setKernelMaxDynamicSharedMemory
std::size_t inline getMaxSharedMemoryPerBlockOptin(std::size_t const dev = 0) {
  int bytes = 0;
  MKN_GPU_ASSERT(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  return bytes;
}

// dynamic shared memory per block kernel f can be launched with on the current device, below
//  the device default when f uses static shared memory, raised by setKernelMaxDynamicSharedMemory
std::size_t inline getKernelMaxDynamicSharedMemory(void const* const f) {
  cudaFuncAttributes attr{};
  MKN_GPU_ASSERT(cudaFuncGetAttributes(&attr, f));
  return attr.maxDynamicSharedSizeBytes;
}

// allows launches of kernel f on the current device with up to bytes of dynamic shared memory
void inline setKernelMaxDynamicSharedMemory(void const* const f, std::size_t const bytes) {
  MKN_GPU_ASSERT(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(bytes)));
}

auto inline getLimitMallocHeapSize() {
  std::size_t bytes = 0;
  MKN_GPU_ASSERT(cudaDeviceGetLimit(&bytes, cudaLimitMallocHeapSize));
  return bytes;
}

void inline setLimitMallocHeapSize(std::size_t const& bytes) {
  MKN_GPU_ASSERT(cudaDeviceSetLimit(cudaLimitMallocHeapSize, bytes));
}

auto inline getLimitStackSize() {
  std::size_t bytes = 0;
  MKN_GPU_ASSERT(cudaDeviceGetLimit(&bytes, cudaLimitStackSize));
  return bytes;
}

void inline setLimitStackSize(std::size_t const& bytes) {
  MKN_GPU_ASSERT(cudaDeviceSetLimit(cudaLimitStackSize, bytes));
}

std::size_t inline getDevice() {
  int dev = 0;
  MKN_GPU_ASSERT(cudaGetDevice(&dev));
  return dev;
}

void inline setDevice(std::size_t const& dev) { MKN_GPU_ASSERT(cudaSetDevice(dev)); }

void inline sync() { MKN_GPU_ASSERT(cudaDeviceSynchronize()); }
void inline sync(cudaStream_t stream) { MKN_GPU_ASSERT(cudaStreamSynchronize(stream)); }

template <typename Size>
void alloc(void*& p, Size size) {
  MKN_GPU_ASSERT(cudaMalloc((void**)&p, size));
}

template <typename T, typename Size>
void alloc(T*& p, Size size) {
  KLOG(TRC) << "GPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(cudaMalloc((void**)&p, size * sizeof(T)));
}

template <typename T, typename Size>
void alloc_host(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(cudaMallocHost((void**)&p, size * sizeof(T)));
}

template <typename T, typename Size>
void alloc_managed(T*& p, Size size) {
  KLOG(TRC) << "GPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(cudaMallocManaged((void**)&p, size * sizeof(T)));
}

void inline destroy(void* p) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaFree(p));
}

template <typename T>
void destroy(T* ptr) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaFree(ptr));
}

template <typename T>
void destroy_host(T* ptr) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaFreeHost(ptr));
}

template <typename T, typename Size>
void copy_on_device(T* dst, T const* src, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename Size>
void send(void* p, void* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpy(p, t, size, cudaMemcpyHostToDevice));
}

template <typename T, typename Size>
void send(T* p, T const* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpy(p, t, size * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T, typename Size>
void send_async(T* p, T const* t, cudaStream_t& stream, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpyAsync(p, t, size * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <typename T, typename Size>
void take(T const* p, T* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpy(t, p, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, typename Size>
void take_async(T const* p, T* t, cudaStream_t& stream, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(cudaMemcpyAsync(t, p, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

// whether the host can access managed memory while kernels run, and managed memory can be
//  prefetched to the device. Without it the host must not touch managed memory while any kernel
//  runs, on any stream
bool inline getConcurrentManagedAccess(std::size_t const dev = 0) {
  int v = 0;
  MKN_GPU_ASSERT(cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentManagedAccess, dev));
  return v != 0;
}

namespace detail {
// cached per thread and device, prefetches can be frequent
bool inline concurrent_managed_access(std::size_t const dev) {
  thread_local std::vector<std::int8_t> cache;  // -1 unknown
  if (cache.size() <= dev) cache.resize(dev + 1, -1);
  if (cache[dev] < 0) cache[dev] = getConcurrentManagedAccess(dev);
  return cache[dev] > 0;
}
}  // namespace detail

// migrate managed memory to device dev ahead of its use there, asynchronous on stream
//  a no-op on devices without concurrent managed access, where prefetching to them fails
template <typename T, typename Size>
void prefetch(T const* p, Size size, std::size_t const dev = 0, cudaStream_t stream = 0) {
  if (!detail::concurrent_managed_access(dev)) return;  // unsupported, it is only a hint
  KLOG(TRC);
#if CUDART_VERSION >= 13000
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id = static_cast<int>(dev);
  MKN_GPU_ASSERT(cudaMemPrefetchAsync(p, size * sizeof(T), location, 0, stream));
#else
  MKN_GPU_ASSERT(cudaMemPrefetchAsync(p, size * sizeof(T), static_cast<int>(dev), stream));
#endif
}

// migrate managed memory to the host ahead of its use there, asynchronous on stream
template <typename T, typename Size>
void prefetch_host(T const* p, Size size, cudaStream_t stream = 0) {
  KLOG(TRC);
#if CUDART_VERSION >= 13000
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeHost;
  MKN_GPU_ASSERT(cudaMemPrefetchAsync(p, size * sizeof(T), location, 0, stream));
#else
  MKN_GPU_ASSERT(cudaMemPrefetchAsync(p, size * sizeof(T), cudaCpuDeviceId, stream));
#endif
}

}  // namespace MKN_GPU_NS

#endif /*MKN_GPU_CUDA_DEF_HPP*/
