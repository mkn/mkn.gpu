#ifndef _MKN_GPU_DEF_HPP_
#define _MKN_GPU_DEF_HPP_

#include <type_traits>

#if !defined(MKN_GPU_ROCM) and __has_include("hip/hip_runtime.h")
#define MKN_GPU_ROCM 1
#endif
#if !defined(MKN_GPU_ROCM)
#define MKN_GPU_ROCM 0
#endif

#if !defined(MKN_GPU_CUDA) and __has_include(<cuda_runtime.h>)
#define MKN_GPU_CUDA 1
#endif
#if !defined(MKN_GPU_CUDA)
#define MKN_GPU_CUDA 0
#endif

#if MKN_GPU_CUDA && MKN_GPU_ROCM && !defined(MKN_GPU_FN_PER_NS)
#define MKN_GPU_FN_PER_NS 1
#endif

#if !defined(MKN_GPU_FN_PER_NS)
#define MKN_GPU_FN_PER_NS 0
#endif

#if MKN_GPU_CUDA == 0 && MKN_GPU_ROCM == 0 && !defined(MKN_GPU_CPU)
#define MKN_GPU_CPU 1
#endif

#if !defined(MKN_GPU_CPU)
#define MKN_GPU_CPU 0
#endif

namespace mkn::gpu {

struct CompileFlags {
  bool constexpr static withCUDA = MKN_GPU_CUDA;
  bool constexpr static withROCM = MKN_GPU_ROCM;
  bool constexpr static withCPU = MKN_GPU_CPU;
  bool constexpr static perNamespace = MKN_GPU_FN_PER_NS;
};

#if MKN_GPU_CPU

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
