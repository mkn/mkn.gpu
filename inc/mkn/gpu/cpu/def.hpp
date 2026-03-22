#ifndef _MKN_GPU_CPU_DEF_HPP_
#define _MKN_GPU_CPU_DEF_HPP_

#include "mkn/kul/log.hpp"
#include "mkn/kul/assert.hpp"

#include <string>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(MKN_GPU_FN_PER_NS) && MKN_GPU_FN_PER_NS
#define MKN_GPU_NS mkn::gpu::cpu
#else
#define MKN_GPU_NS mkn::gpu
#endif  // MKN_GPU_FN_PER_NS

#if defined(__device__)
#pragma message("__device__ already defined")
#error  // check your compiler
#endif

#if defined(__host__)
#pragma message("__host__ already defined")
#error  // check your compiler
#endif

#if defined(__global__)
#pragma message("__global__ already defined")
#error  // check your compiler
#endif

// we need to exclude these for CPU only operations
#define __shared__
#define __device__
#define __host__
#define __global__
#define __syncthreads(...)

#if !defined(MKN_CPU_DO_NOT_DEFINE_DIM3)
#define MKN_CPU_DO_NOT_DEFINE_DIM3 0
#endif

#if !defined(dim3) and !MKN_CPU_DO_NOT_DEFINE_DIM3
struct dim3 {
  dim3() {}
  dim3(std::size_t x) : x{x} {}
  dim3(std::size_t x, std::size_t y) : x{x}, y{y} {}
  dim3(std::size_t x, std::size_t y, std::size_t z) : x{x}, y{y}, z{z} {}

  std::size_t x = 1, y = 1, z = 1;
};
dim3 static inline threadIdx, blockIdx;
#endif  // MKN_CPU_DO_NOT_DEFINE_DIM3

namespace MKN_GPU_NS {

#define MKN_GPU_ASSERT(x) (KASSERT((x)))

struct DeviceProperties {
  std::string major = "major";
  std::string minor = "minor";
  std::string name = "name";
  std::size_t multiProcessorCount = 1;
  std::size_t maxThreadsPerMultiProcessor = 0;
  std::size_t totalGlobalMem = 0;
  std::size_t sharedMemPerBlock = 0;
  std::size_t warpSize = 1;
  std::size_t maxThreadsPerBlock = 0;
};

auto inline getDeviceProperties(std::size_t /*dev*/ = 0) { return DeviceProperties{}; }

std::uint32_t inline getWarpSize(size_t /*dev */ = 0) { return 1; }

static std::uint32_t inline const warp_size = getWarpSize();

auto inline getLimitMallocHeapSize() {
  std::size_t bytes = 0;

  return bytes;
}

void inline setLimitMallocHeapSize(std::size_t const& /*bytes*/) {}

void inline setDevice(std::size_t const& /*dev*/) {} /*noop*/

template <typename Size>
void alloc(void*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size;
  MKN_GPU_ASSERT(p = std::malloc(size));
}

template <typename T, typename Size>
void alloc(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(p = reinterpret_cast<T*>(std::malloc(size * sizeof(T))));
}

template <typename T, typename Size>
void alloc_host(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(p = reinterpret_cast<T*>(std::malloc(size * sizeof(T))));
}

template <typename T, typename Size>
void alloc_managed(T*& p, Size size) {
  KLOG(TRC) << "CPU alloced: " << size * sizeof(T);
  MKN_GPU_ASSERT(p = reinterpret_cast<T*>(std::malloc(size * sizeof(T))));
}

void inline destroy(void* p) {
  KLOG(TRC);
  std::free(p);
}

template <typename T>
void destroy(T* p) {
  KLOG(TRC);
  std::free(p);
}

template <typename T>
void destroy_host(T* p) {
  KLOG(TRC);
  std::free(p);
}

template <typename T, typename Size>
void copy_on_device(T* dst, T const* src, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(std::memcpy(dst, src, size * sizeof(T)));
}

template <typename Size>
void send(void* p, void* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(std::memcpy(p, t, size));
}

template <typename T, typename Size>
void send(T* p, T const* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(std::memcpy(p, t, size * sizeof(T)));
}
template <typename T, typename Size>
void send_async(T* p, T const* t, auto& /*stream*/, Size size = 1) {
  KLOG(TRC);
  send(p, t, size);
}

template <typename T, typename Size>
void take(T const* p, T* t, Size size = 1) {
  KLOG(TRC);
  MKN_GPU_ASSERT(std::memcpy(t, p, size * sizeof(T)));
}

template <typename T, typename Size>
void take_async(T const* p, T* t, auto& /*stream*/, Size size = 1) {
  KLOG(TRC);
  take(p, t, size);
}

void inline sync() {}

}  // namespace MKN_GPU_NS

#endif /*_MKN_GPU_CPU_DEF_HPP_*/
