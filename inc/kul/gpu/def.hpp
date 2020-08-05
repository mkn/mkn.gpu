

#ifndef _KUL_GPU_DEF_HPP_
#define _KUL_GPU_DEF_HPP_

#include <type_traits>

#if defined(KUL_GPU_ROCM)
#include "kul/gpu/rocm/def.hpp"
#elif defined(KUL_GPU_CUDA)
#include "kul/gpu/cuda/def.hpp"
#elif !defined(KUL_GPU_FN_PER_NS) || KUL_GPU_FN_PER_NS == 0
#error "UNKNOWN GPU / define KUL_GPU_ROCM or KUL_GPU_CUDA"
#endif

namespace kul::gpu {

template <typename T>
static constexpr bool is_floating_point_v =
    std::is_floating_point_v<T> or std::is_same_v<_Float16, T>;

} /* namespace kul::gpu */

#endif /*_KUL_GPU_DEF_HPP_*/
