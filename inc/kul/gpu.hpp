/**
Copyright (c) 2019, Philip Deegan.
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
#ifndef _KUL_GPU_HPP_
#define _KUL_GPU_HPP_

#if defined(KUL_GPU_ROCM)
#include "kul/gpu/rocm.hpp"
#elif defined(KUL_GPU_CUDA)
#include "kul/gpu/cuda.hpp"
#else
#error   "UNKNOWN GPU / define KUL_GPU_ROCM or KUL_GPU_CUDA"
#endif

namespace kul::gpu {

__device__ uint32_t idx() {

#if defined(KUL_GPU_ROCM)
  return kul::gpu::hip::idx();
#elif defined(KUL_GPU_CUDA)
  return kul::gpu::cuda::idx();
#else
#error   "UNKNOWN GPU / define KUL_GPU_ROCM or KUL_GPU_CUDA"
#endif

}

} /* namespace kul::gpu */


#endif /* _KUL_GPU_HPP_ */
