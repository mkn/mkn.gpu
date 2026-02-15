#ifndef _MKN_GPU_ROCM_API_HPP_
#define _MKN_GPU_ROCM_API_HPP_

#include <cstdint>

#include "hip/hip_runtime.h"

#include "mkn/gpu/rocm/def.hpp"

namespace mkn::gpu::hip {

template <typename SIZE = std::uint32_t /*max 4294967296*/>
__device__ SIZE idx() {
  SIZE width = gridDim.x * blockDim.x;
  SIZE height = gridDim.y * blockDim.y;
  SIZE x = blockDim.x * blockIdx.x + threadIdx.x;
  SIZE y = blockDim.y * blockIdx.y + threadIdx.y;
  SIZE z = blockDim.z * blockIdx.z + threadIdx.z;
  return x + (y * width) + (z * width * height);
}

template <typename SIZE = std::uint32_t /*max 4294967296*/>
__device__ SIZE block_idx_x() {
  return blockIdx.x;
}

}  // namespace mkn::gpu::hip

namespace MKN_GPU_NS {

template <typename F, typename... Args>
__global__ static void global_gd_kernel(F f, std::size_t s, Args... args) {
  if (auto i = mkn::gpu::hip::idx(); i < s) f(args...);
}

template <typename F, typename... Args>
__global__ static void global_d_kernel(F f, Args... args) {
  f(args...);
}

}  // namespace MKN_GPU_NS

//

#endif /*_MKN_GPU_ROCM_API_HPP_*/
