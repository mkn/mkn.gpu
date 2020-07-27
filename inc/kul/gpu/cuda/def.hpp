

#ifndef _KUL_GPU_CUDA_DEF_HPP_
#define _KUL_GPU_CUDA_DEF_HPP_

namespace kul::gpu::cuda {

__device__ uint32_t idx() {
  uint32_t width = gridDim.x * blockDim.x;
  uint32_t height = gridDim.y * blockDim.y;

  uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t z = blockDim.z * blockIdx.z + threadIdx.z;
  return x + (y * width) + (z * width * height);  // max 4294967296
}

}  // namespace kul::gpu::cuda

#endif /*_KUL_GPU_CUDA_DEF_HPP_*/
