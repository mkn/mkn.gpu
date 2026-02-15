#ifndef _MKN_GPU_ANY_CLS_HPP_
#define _MKN_GPU_ANY_CLS_HPP_

#include "mkn/kul/env.hpp"
#include "mkn/kul/string.hpp"

namespace mkn::gpu {

template <typename Device>
struct Cli {
  constexpr static inline char const* MKN_GPU_BX_THREADS = "MKN_GPU_BX_THREADS";

  auto bx_threads() const {
    if (kul::env::EXISTS(MKN_GPU_BX_THREADS))
      return kul::String::INT32(kul::env::GET(MKN_GPU_BX_THREADS));
    return dev.maxThreadsPerBlock;
  }

  Device const& dev;
};

} /* namespace mkn::gpu */

#endif /*_MKN_GPU_ANY_CLS_HPP_*/
