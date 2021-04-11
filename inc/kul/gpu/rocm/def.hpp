
// IWYU pragma: private, include "kul/gpu/def.hpp"

#ifndef _KUL_GPU_ROCM_DEF_HPP_
#define _KUL_GPU_ROCM_DEF_HPP_

#include "hip/hip_runtime.h"

namespace kul::gpu::hip {

template <typename SIZE = uint32_t /*max 4294967296*/>
__device__ SIZE idx() {
  SIZE width = hipGridDim_x * hipBlockDim_x;
  SIZE height = hipGridDim_y * hipBlockDim_y;

  SIZE x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  SIZE y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  SIZE z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  return x + (y * width) + (z * width * height);  // max 4294967296
}

}  // namespace kul::gpu::hip

#endif /*_KUL_GPU_ROCM_DEF_HPP_*/
