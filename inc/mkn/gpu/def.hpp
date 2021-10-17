

#ifndef _MKN_GPU_DEF_HPP_
#define _MKN_GPU_DEF_HPP_

#include <type_traits>

#if defined(MKN_GPU_ROCM)
#include "mkn/gpu/rocm/def.hpp"
#elif defined(MKN_GPU_CUDA)
#include "mkn/gpu/cuda/def.hpp"
#elif !defined(MKN_GPU_FN_PER_NS) || MKN_GPU_FN_PER_NS == 0
#error "UNKNOWN GPU / define MKN_GPU_ROCM or MKN_GPU_CUDA"
#endif

namespace mkn::gpu {

template <typename T>
static constexpr bool is_floating_point_v =
    std::is_floating_point_v<T> or std::is_same_v<_Float16, T>;

} /* namespace mkn::gpu */

#endif /*_MKN_GPU_DEF_HPP_*/
