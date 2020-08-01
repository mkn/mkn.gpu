

#ifndef _KUL_GPU_DEF_HPP_
#define _KUL_GPU_DEF_HPP_


#if defined(KUL_GPU_ROCM)
#include "kul/gpu/rocm.hpp"
#elif defined(KUL_GPU_CUDA)
#include "kul/gpu/cuda.hpp"
#else
#error "UNKNOWN GPU / define KUL_GPU_ROCM or KUL_GPU_CUDA"
#endif


namespace kul::gpu {

template <typename T>
static constexpr bool is_floating_point_v =
    std::is_floating_point_v<T> or std::is_same_v<_Float16, T>;

__device__ uint32_t idx() {
#if defined(KUL_GPU_ROCM)
  return kul::gpu::hip::idx();
#elif defined(KUL_GPU_CUDA)
  return kul::gpu::cuda::idx();
#else
#error "UNKNOWN GPU / define KUL_GPU_ROCM or KUL_GPU_CUDA"
#endif
}

} /* namespace kul::gpu */

#endif /*_KUL_GPU_DEF_HPP_*/
