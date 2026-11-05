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
#ifndef MKN_GPU_TRACE_HPP
#define MKN_GPU_TRACE_HPP

// Profiler annotations, visible in Nsight Systems (NVTX) or rocprofv3 (roctx)
//
//  MKN_GPU_TRACE=0 disables all annotations
//  CUDA: on by default, NVTX3 is header only
//  ROCm: off by default, enable with -DMKN_GPU_TRACE=1 and link
//        -lrocprofiler-sdk-roctx (or -lroctx64 for the legacy roctracer header)
//
//  nsys profile --trace=cuda,nvtx ./app
//  rocprofv3 --marker-trace --kernel-trace -- ./app

#include "mkn/gpu.hpp"

#include <cstdint>

#if !defined(MKN_GPU_TRACE)
#if MKN_GPU_CUDA && __has_include(<nvtx3/nvToolsExt.h>)
#define MKN_GPU_TRACE 1
#else
#define MKN_GPU_TRACE 0
#endif
#endif

#if MKN_GPU_TRACE && MKN_GPU_CUDA
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#elif MKN_GPU_TRACE && MKN_GPU_ROCM
#if __has_include(<rocprofiler-sdk-roctx/roctx.h>)
#include <rocprofiler-sdk-roctx/roctx.h>
#else
#include <roctracer/roctx.h>
#endif
#endif

namespace mkn::gpu::trace {

bool constexpr static enabled = MKN_GPU_TRACE;

#if MKN_GPU_TRACE && MKN_GPU_CUDA

namespace detail {
nvtxDomainHandle_t inline domain() {
  static nvtxDomainHandle_t const d = nvtxDomainCreateA("mkn.gpu");
  return d;
}
}  // namespace detail

// payload shows as a number on the range, e.g. the data/stream index
void inline push(char const* const name, std::int64_t const payload = -1) {
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = name;
  if (payload >= 0) {
    attr.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    attr.payload.llValue = payload;
  }
  nvtxDomainRangePushEx(detail::domain(), &attr);
}
void inline pop() { nvtxDomainRangePop(detail::domain()); }
void inline mark(char const* const name) {
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = name;
  nvtxDomainMarkEx(detail::domain(), &attr);
}
template <typename Stream>
void name_stream(Stream& stream, char const* const name) {
  nvtxNameCudaStreamA(stream(), name);
}

#elif MKN_GPU_TRACE && MKN_GPU_ROCM

void inline push(char const* const name, std::int64_t const = -1) { roctxRangePush(name); }
void inline pop() { roctxRangePop(); }
void inline mark(char const* const name) { roctxMark(name); }
template <typename Stream>
void name_stream(Stream&, char const* const) {}

#else

void inline push(char const* const, std::int64_t const = -1) {}
void inline pop() {}
void inline mark(char const* const) {}
template <typename Stream>
void name_stream(Stream&, char const* const) {}

#endif

// RAII range, nests per host thread
struct Range {
  Range(char const* const name, std::int64_t const payload = -1) { push(name, payload); }
  ~Range() { pop(); }
  Range(Range const&) = delete;
  Range& operator=(Range const&) = delete;
};

}  // namespace mkn::gpu::trace

#endif /* MKN_GPU_TRACE_HPP */
