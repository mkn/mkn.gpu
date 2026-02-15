#ifndef _MKN_GPU_CUDA_DEF_HPP_
#define _MKN_GPU_CUDA_DEF_HPP_

#include <string>
#include <cstdint>

#include <cuda_runtime.h>

#include "mkn/kul/log.hpp"
#include "mkn/gpu/def.hpp"

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

std::uint32_t inline getWarpSize(size_t dev = 0) {
#ifdef _MKN_GPU_WARP_SIZE_
  return _MKN_GPU_WARP_SIZE_;
#else
  cudaDeviceProp devProp;
  [[maybe_unused]] auto ret = cudaGetDeviceProperties(&devProp, dev);
  return devProp.warpSize;
#endif /*_MKN_GPU_WARP_SIZE_    */
}

static std::uint32_t inline const warp_size = getWarpSize();

void inline setLimitMallocHeapSize(std::size_t const& bytes) {
  MKN_GPU_ASSERT(cudaDeviceSetLimit(cudaLimitMallocHeapSize, bytes));
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

}  // namespace MKN_GPU_NS

#endif /*_MKN_GPU_CUDA_DEF_HPP_*/
