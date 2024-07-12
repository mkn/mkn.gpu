

#ifndef _MKN_GPU_DEFINES_HPP_
#define _MKN_GPU_DEFINES_HPP_

#include <type_traits>

#if !defined(MKN_GPU_FN_PER_NS)
#define MKN_GPU_FN_PER_NS 0
#endif

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

#if MKN_GPU_CUDA == 1 && MKN_GPU_ROCM == 1 && MKN_GPU_FN_PER_NS == 0
#define MKN_GPU_FN_PER_NS 1
#endif

#if MKN_GPU_ROCM == 1
#include "mkn/gpu/rocm.hpp"
#endif

#if MKN_GPU_CUDA
#include "mkn/gpu/cuda.hpp"
#endif

#if MKN_GPU_FN_PER_NS == 1 || MKN_GPU_CPU == 1
#include "mkn/gpu/cpu.hpp"
#endif

namespace mkn::gpu {

struct CompileFlags {
  bool constexpr static withCUDA = MKN_GPU_CUDA;
  bool constexpr static withROCM = MKN_GPU_ROCM;
  bool constexpr static withCPU = !MKN_GPU_ROCM and !MKN_GPU_CUDA;
  bool constexpr static perNamespace = MKN_GPU_FN_PER_NS;
};

} /* namespace mkn::gpu */

#endif /*_MKN_GPU_DEFINES_HPP_*/
