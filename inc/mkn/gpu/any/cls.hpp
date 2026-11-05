#ifndef _MKN_GPU_ANY_CLS_HPP_
#define _MKN_GPU_ANY_CLS_HPP_

#include <limits>
#include <string>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace mkn::gpu {

// block x dimension for a guessed launch of a specific kernel, the cap (device maximum) limited
//  by the kernel's own maximum, which is rounded down to a warp multiple
std::size_t inline clamp_bx_threads(std::size_t const cap, std::size_t const kernel_max,
                                    std::size_t const warp) {
  if (cap <= kernel_max) return cap;
  if (warp > 1 && kernel_max >= warp) return kernel_max - (kernel_max % warp);
  return kernel_max;
}

// grid dimension for guessed launches covering size with blocks of bx, the launch must fit in 32
//  bits, idx() is 32 bit by default and HIP limits a launch to 2^32 - 1 threads
std::size_t inline guess_grid_dim(std::size_t const size, std::size_t const bx) {
  auto const blocks = size / bx + (size % bx > 0);
  if (blocks * bx > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("mkn.gpu error: guessed launch size " + std::to_string(size) +
                                " with block size " + std::to_string(bx) +
                                " exceeds 32 bit indexing");
  return blocks;
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
