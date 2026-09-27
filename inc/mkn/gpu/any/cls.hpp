#ifndef _MKN_GPU_ANY_CLS_HPP_
#define _MKN_GPU_ANY_CLS_HPP_

#include "mkn/kul/env.hpp"
#include "mkn/kul/string.hpp"

#include <string>
#include <cstddef>
#include <stdexcept>

namespace mkn::gpu {

constexpr static inline char const* MKN_GPU_BX_THREADS = "MKN_GPU_BX_THREADS";

// block x dimension for guessed launches, environment override or device maximum
std::size_t inline bx_threads(std::size_t const max_threads_per_block) {
  if (kul::env::EXISTS(MKN_GPU_BX_THREADS))
    return kul::String::INT32(kul::env::GET(MKN_GPU_BX_THREADS));
  return max_threads_per_block;
}

// grid dimension for fixed launches, the problem size must be a multiple of the block size
std::size_t inline grid_dim(std::size_t const size, std::size_t const threads) {
  if (threads == 0 || size % threads > 0)
    throw std::invalid_argument("mkn.gpu error: launch size " + std::to_string(size) +
                                " is not a multiple of block size " + std::to_string(threads));
  return size / threads;
}

} /* namespace mkn::gpu */

#endif /*_MKN_GPU_ANY_CLS_HPP_*/
