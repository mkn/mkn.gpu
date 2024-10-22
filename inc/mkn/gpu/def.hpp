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

#ifndef _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_
#define _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_ 1
#endif /*_MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_    */

#ifndef _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_ADD_
#define _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_ADD_ 1
#endif /*_MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_ADD_    */

#ifndef _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_MAX_
#define _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_MAX_ 25
#endif /*_MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_MAX_    */

} /* namespace mkn::gpu */

#endif /*_MKN_GPU_DEF_HPP_*/
