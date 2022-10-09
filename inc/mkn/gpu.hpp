/**
Copyright (c) 2020, Philip Deegan.
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
#ifndef _MKN_GPU_HPP_
#define _MKN_GPU_HPP_

#if defined(MKN_GPU_ROCM)
#include "mkn/gpu/rocm.hpp"
#elif defined(MKN_GPU_CUDA)
#include "mkn/gpu/cuda.hpp"
#elif defined(MKN_GPU_CPU)
#include "mkn/gpu/cpu.hpp"
#elif !defined(MKN_GPU_FN_PER_NS) || MKN_GPU_FN_PER_NS == 0
#error "UNKNOWN GPU / define MKN_GPU_ROCM or MKN_GPU_CUDA"
#endif

namespace mkn::gpu {

__device__ uint32_t idx() {
#if defined(MKN_GPU_ROCM)
  return mkn::gpu::hip::idx();
#elif defined(MKN_GPU_CUDA)
  return mkn::gpu::cuda::idx();
#elif defined(MKN_GPU_CPU)
  return mkn::gpu::cpu::idx();
#else
#error "UNKNOWN GPU / define MKN_GPU_ROCM or MKN_GPU_CUDA"
#endif
}

} /* namespace mkn::gpu */

#endif /* _MKN_GPU_HPP_ */
