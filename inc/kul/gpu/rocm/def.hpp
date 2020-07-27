

#ifndef _KUL_GPU_ROCM_DEF_HPP_
#define _KUL_GPU_ROCM_DEF_HPP_

namespace kul::gpu::hip {

__device__ uint32_t idx() {
  uint32_t width = hipGridDim_x * hipBlockDim_x;
  uint32_t height = hipGridDim_y * hipBlockDim_y;

  uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  uint32_t z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  return x + (y * width) + (z * width * height);  // max 4294967296
}

}  // namespace kul::gpu::hip

#endif /*_KUL_GPU_ROCM_DEF_HPP_*/
