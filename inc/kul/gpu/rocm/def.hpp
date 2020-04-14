

#ifndef _KUL_GPU_ROCM_DEF_HPP_
#define _KUL_GPU_ROCM_DEF_HPP_

namespace kul::gpu::hip {

__device__ int blockDim_i(size_t width, size_t height){
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  int z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  return x  + (y * width) + (z * height);
}

}

#endif /*_KUL_GPU_ROCM_DEF_HPP_*/
