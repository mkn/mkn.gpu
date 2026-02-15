/**
Copyright (c) 2024, Philip Deegan.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Philip Deegan nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// IWYU pragma: private, include "mkn/gpu.hpp"
#ifndef _MKN_GPU_ROCM_HPP_
#define _MKN_GPU_ROCM_HPP_

#include "mkn/kul/log.hpp"

#include "hip/hip_runtime.h"

#include "mkn/gpu/rocm/def.hpp"
#include "mkn/gpu/rocm/api.hpp"
#include "mkn/gpu/rocm/cls.hpp"

namespace MKN_GPU_NS {

template <typename T, typename V>
__global__ void _vector_fill(T* a, V t, std::size_t s) {
  if (auto i = mkn::gpu::hip::idx(); i < s) a[i] = t;
}

template <typename Container, typename T>
void fill(Container& c, std::size_t const size, T const val) {
  GLauncher{c.size()}(_vector_fill<typename Container::value_type, T>, c.data(), val, size);
}

template <typename Container, typename T>
void fill(Container& c, T const val) {
  fill(c, c.size(), val);
}

// https://rocm-developer-tools.github.io/HIP/group__Device.html
void inline prinfo(size_t dev = 0) {
  hipDeviceProp_t devProp;
  MKN_GPU_ASSERT(hipGetDeviceProperties(&devProp, dev));
  KOUT(NON) << " System version  " << devProp.major << "." << devProp.minor;
  KOUT(NON) << " agent name      " << devProp.name;
  KOUT(NON) << " cores           " << devProp.multiProcessorCount;
  KOUT(NON) << " threadsPCore    " << devProp.maxThreadsPerMultiProcessor;
  KOUT(NON) << " TotalMem        " << (devProp.totalGlobalMem / 1000000) << " MB";
  KOUT(NON) << " BlockMem        " << (devProp.sharedMemPerBlock / 1000) << " KB";
  KOUT(NON) << " device warpSize " << devProp.warpSize;
  KOUT(NON) << " threadsPBlock   " << devProp.maxThreadsPerBlock;

#ifdef _MKN_GPU_WARP_SIZE_
  KOUT(NON) << " warpSize used   " << _MKN_GPU_WARP_SIZE_;
#else
  KOUT(NON) << " warpSize used   " << warp_size;
  if (warp_size != static_cast<std::uint32_t>(devProp.warpSize)) {
    KOUT(NON) << " warpSize MISMATCH!!!  " << warp_size << " vs " << devProp.warpSize;
    KOUT(NON) << " SEE mkn.gpu README for -D_MKN_GPU_WARP_SIZE_=###";
  }
#endif
}

void inline print_gpu_mem_used() {
  float free_m = 0, total_m = 0, used_m = 0;
  std::size_t free_t = 0, total_t = 0;
  MKN_GPU_ASSERT(hipMemGetInfo(&free_t, &total_t));
  free_m = free_t / 1048576.0;
  total_m = total_t / 1048576.0;
  used_m = total_m - free_m;
  printf("  mem free %zu .... %f MB mem total %zu....%f MB mem used %f MB\n", free_t, free_m,
         total_t, total_m, used_m);
}

#include "mkn/gpu/any/inc/launchers.ipp"
#include "mkn/gpu/any/inc/devfunc.ipp"

}  // namespace MKN_GPU_NS

#undef MKN_GPU_ASSERT

#endif /* _MKN_GPU_ROCM_HPP_ */
