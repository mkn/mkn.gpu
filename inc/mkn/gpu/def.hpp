#ifndef MKN_GPU_DEF_HPP
#define MKN_GPU_DEF_HPP

#include "mkn/kul/lang/cpp/compiler/def/inline.hpp"  // IWYU pragma: export

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#if MKN_GPU_ROCM and MKN_GPU_CUDA
#error  // not possible
#endif

#if MKN_GPU_ROCM
#define MKN_GPU_CUDA 0
#endif

#if MKN_GPU_CUDA
#define MKN_GPU_ROCM 0
#endif

#if MKN_GPU_CPU
#define MKN_GPU_ROCM 0
#define MKN_GPU_CUDA 0
#endif

#if !defined(MKN_GPU_ROCM) and !defined(MKN_GPU_CUDA)
#if __has_include(<cuda_runtime.h>) and __has_include("hip/hip_runtime.h")
#error  // ACTIVATE ONE!
#endif  // __has_include(...)
#endif  // !defined(...)

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

// mkn.kul functions that can also run on a device are __host__ __device__ with a GPU backend,
//  mkn.kul leaves MKN_KUL_FN_SIG_ALL as host only unless it's defined first, so include mkn.gpu
//  before mkn.kul
#if !defined(MKN_KUL_FN_SIG_ALL) && (MKN_GPU_CUDA || MKN_GPU_ROCM)
#define MKN_KUL_FN_SIG_ALL __host__ __device__
#endif

namespace mkn::gpu {

struct CompileFlags {
  bool constexpr static withCUDA = MKN_GPU_CUDA;
  bool constexpr static withROCM = MKN_GPU_ROCM;
  bool constexpr static withCPU = MKN_GPU_CPU;
  bool constexpr static perNamespace = MKN_GPU_FN_PER_NS;
};

// a device's properties, the same for every backend (getDeviceProperties), the fields mkn.gpu
//  uses of CUDA's cudaDeviceProp / HIP's hipDeviceProp_t, which differ between backends and
//  versions. warpSize is the width this build's kernels run at on the device, from a kernel: it
//  can differ from the hardware's own (AMD RDNA is wave32 unless compiled -mwavefrontsize64)
struct DeviceProperties {
  std::string name;
  std::string runtime;  // backend and its version
  std::uint32_t major = 0, minor = 0;
  std::size_t multiProcessorCount = 0;
  std::size_t maxThreadsPerMultiProcessor = 0;
  std::size_t maxThreadsPerBlock = 0;
  std::size_t totalGlobalMem = 0;
  std::size_t sharedMemPerBlock = 0;
  std::uint32_t warpSize = 0;
};

#if MKN_GPU_CPU

template <typename T>
static constexpr bool is_floating_point_v = std::is_floating_point_v<T>;

#else

template <typename T>
static constexpr bool is_floating_point_v =
    std::is_floating_point_v<T> or std::is_same_v<_Float16, T>;

#endif

} /* namespace mkn::gpu */

#endif /*MKN_GPU_DEF_HPP*/
