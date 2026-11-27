// IWYU pragma: private, include "mkn/gpu.hpp"
// IWYU pragma: friend ".*mkn/gpu/(cpu|cuda|rocm|any)(/.*|\.hpp)"

#ifndef MKN_GPU_ROCM_DEF_HPP
#define MKN_GPU_ROCM_DEF_HPP

#include "mkn/kul/fail.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/gpu/def.hpp"
#include "hip/hip_runtime.h"

#include <string>
#include <cstdint>
#include <map>
#include <source_location>
#include <mutex>
#include <utility>
#include <vector>

#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
#define MKN_GPU_NS mkn::gpu::hip
#else
#define MKN_GPU_NS mkn::gpu
#endif  // MKN_GPU_FN_PER_NS

namespace MKN_GPU_NS {

static_assert(CompileFlags::withROCM);

// an API error is unrecoverable: mkn::kul::fail, where the macro is used
#define MKN_GPU_ASSERT(ans) \
  {                         \
    gpuAssert((ans));       \
  }
inline void gpuAssert(hipError_t const code,
                      std::source_location const where = std::source_location::current()) {
  if (code != hipSuccess)
    mkn::kul::fail(std::string{"GPUassert: "} + hipGetErrorString(code), where);
}

std::string getErrorString(auto const code) { return hipGetErrorString(code); }

namespace detail {
__global__ static void warp_size_probe(std::uint32_t* const out) { *out = warpSize; }

// the warpSize this build's kernels see on device dev: the code object's wavefront size, not
//  hipDeviceProp_t::warpSize, the hardware default (wave32 on RDNA, whatever the code object)
std::uint32_t inline probe_warp_size(std::size_t const dev) {
  int current = 0;
  MKN_GPU_ASSERT(hipGetDevice(&current));
  MKN_GPU_ASSERT(hipSetDevice(static_cast<int>(dev)));
  std::uint32_t* d = nullptr;
  MKN_GPU_ASSERT(hipMalloc(reinterpret_cast<void**>(&d), sizeof(*d)));
  warp_size_probe<<<1, 1>>>(d);
  MKN_GPU_ASSERT(hipGetLastError());
  std::uint32_t w = 0;
  MKN_GPU_ASSERT(hipMemcpy(&w, d, sizeof(w), hipMemcpyDeviceToHost));
  MKN_GPU_ASSERT(hipFree(d));
  MKN_GPU_ASSERT(hipSetDevice(current));
  return w;
}
}  // namespace detail

// device dev's properties, queried once per device. A device whose properties cannot be read,
//  or that cannot run a kernel, is an invalid setup: mkn::kul::fail
DeviceProperties inline const& getDeviceProperties(std::size_t const dev = 0) {
  static std::mutex mutex;
  static std::map<std::size_t, DeviceProperties> cache;
  std::lock_guard<std::mutex> lock{mutex};
  if (auto const it = cache.find(dev); it != cache.end()) return it->second;

  hipDeviceProp_t p;
  MKN_GPU_ASSERT(hipGetDeviceProperties(&p, static_cast<int>(dev)));
  DeviceProperties props;
  props.name = p.name;
  props.runtime = "HIP " + std::to_string(HIP_VERSION_MAJOR) + "." +
                  std::to_string(HIP_VERSION_MINOR) + "." + std::to_string(HIP_VERSION_PATCH);
  props.major = static_cast<std::uint32_t>(p.major);
  props.minor = static_cast<std::uint32_t>(p.minor);
  props.multiProcessorCount = static_cast<std::size_t>(p.multiProcessorCount);
  props.maxThreadsPerMultiProcessor = static_cast<std::size_t>(p.maxThreadsPerMultiProcessor);
  props.maxThreadsPerBlock = static_cast<std::size_t>(p.maxThreadsPerBlock);
  props.totalGlobalMem = p.totalGlobalMem;
  props.sharedMemPerBlock = p.sharedMemPerBlock;
  props.warpSize = detail::probe_warp_size(dev);
  if (props.warpSize == 0) mkn::kul::fail("no warpSize from device " + std::to_string(dev));
  return cache.emplace(dev, std::move(props)).first->second;
}

std::uint32_t inline getWarpSize(std::size_t const dev = 0) {
  return getDeviceProperties(dev).warpSize;
}

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

// dynamic shared memory per block kernel f can be launched with on the current device, below
//  the device default when f uses static shared memory, raised by setKernelMaxDynamicSharedMemory
std::size_t inline getKernelMaxDynamicSharedMemory(void const* const f) {
  hipFuncAttributes attr{};
  MKN_GPU_ASSERT(hipFuncGetAttributes(&attr, f));
  return attr.maxDynamicSharedSizeBytes;
}

// static shared memory per block of kernel f (its __shared__ arrays), taken from the opt-in maximum
std::size_t inline getKernelStaticSharedMemory(void const* const f) {
  hipFuncAttributes attr{};
  MKN_GPU_ASSERT(hipFuncGetAttributes(&attr, f));
  return attr.sharedSizeBytes;
}

// allows launches of kernel f on the current device with up to bytes of dynamic shared memory
void inline setKernelMaxDynamicSharedMemory(void const* const f, std::size_t const bytes) {
  MKN_GPU_ASSERT(
      hipFuncSetAttribute(f, hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
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

std::size_t inline getDevice() {
  int dev = 0;
  MKN_GPU_ASSERT(hipGetDevice(&dev));
  return dev;
}

void inline setDevice(std::size_t const& dev) { MKN_GPU_ASSERT(hipSetDevice(dev)); }

// the warpSize kernels run at on device dev (see DeviceProperties), cached per thread and device
std::uint32_t inline warp_size(std::size_t const dev) {
  thread_local std::vector<std::uint32_t> cache;
  if (cache.size() <= dev) cache.resize(dev + 1, 0);
  if (cache[dev] == 0) cache[dev] = getWarpSize(dev);
  return cache[dev];
}

// on the current device
std::uint32_t inline warp_size() { return warp_size(getDevice()); }

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

// as copy_on_device, queued on the default stream (no stream at hand, e.g. a lone buffer's
//  regrow): dst ready after a sync, or for anything later on the default stream
template <typename T, typename Size = std::size_t>
void copy_on_device_async(T* dst, T const* src, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemcpyAsync(dst, src, size * sizeof(T), hipMemcpyDeviceToDevice, 0));
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

// size values src -> dst queued on stream, any pointers the runtime knows (host, device, managed:
//  hipMemcpyDefault picks the direction); dst ready after the stream's sync
template <typename T, typename Size>
void copy_async(T* dst, T const* src, hipStream_t& stream, Size size) {
  KLOG(TRC);
  MKN_GPU_ASSERT(hipMemcpyAsync(dst, src, size * sizeof(T), hipMemcpyDefault, stream));
}

// whether the host can access managed memory while kernels run, and managed memory can be
//  prefetched to the device. Without it the host must not touch managed memory while any kernel
//  runs, on any stream
bool inline getConcurrentManagedAccess(std::size_t const dev = 0) {
  int v = 0;
  MKN_GPU_ASSERT(hipDeviceGetAttribute(&v, hipDeviceAttributeConcurrentManagedAccess, dev));
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
void prefetch(T const* p, Size size, std::size_t const dev = 0, hipStream_t stream = 0) {
  if (!detail::concurrent_managed_access(dev)) return;  // unsupported, it is only a hint
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

#endif /*MKN_GPU_ROCM_DEF_HPP*/
