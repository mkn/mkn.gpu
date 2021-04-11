
// IWYU pragma: private, include "kul/gpu/def.hpp"

#ifndef _KUL_GPU_CUDA_DEF_HPP_
#define _KUL_GPU_CUDA_DEF_HPP_

#include <cuda_runtime.h>

namespace kul::gpu::cuda {

template <typename SIZE = uint32_t /*max 4294967296*/>
__device__ SIZE idx() {
  SIZE width = gridDim.x * blockDim.x;
  SIZE height = gridDim.y * blockDim.y;

  SIZE x = blockDim.x * blockIdx.x + threadIdx.x;
  SIZE y = blockDim.y * blockIdx.y + threadIdx.y;
  SIZE z = blockDim.z * blockIdx.z + threadIdx.z;
  return x + (y * width) + (z * width * height);
}

}  // namespace kul::gpu::cuda

#endif /*_KUL_GPU_CUDA_DEF_HPP_*/
