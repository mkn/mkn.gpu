#ifndef _MKN_GPU_DEF_HPP_
#define _MKN_GPU_DEF_HPP_

#include <type_traits>

namespace mkn::gpu {

#if defined(MKN_GPU_CPU)

template <typename T>
static constexpr bool is_floating_point_v = std::is_floating_point_v<T>;

#else

template <typename T>
static constexpr bool is_floating_point_v =
    std::is_floating_point_v<T> or std::is_same_v<_Float16, T>;

#endif

} /* namespace mkn::gpu */

#endif /*_MKN_GPU_DEF_HPP_*/
