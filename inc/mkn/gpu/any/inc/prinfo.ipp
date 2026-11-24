// IWYU pragma: private, include "mkn/gpu.hpp"
// IWYU pragma: friend ".*mkn/gpu/(cpu|cuda|rocm|any)(/.*|\.hpp)"

#ifndef MKN_GPU_ANY_INC_PRINFO_IPP
#define MKN_GPU_ANY_INC_PRINFO_IPP

// device dev's properties (getDeviceProperties), the same for every backend
void inline prinfo(std::size_t const dev = 0) {
  auto const& p = getDeviceProperties(dev);
  KOUT(NON) << " System version  " << p.major << "." << p.minor;
  KOUT(NON) << " agent name      " << p.name;
  KOUT(NON) << " cores           " << p.multiProcessorCount;
  KOUT(NON) << " threadsPCore    " << p.maxThreadsPerMultiProcessor;
  KOUT(NON) << " TotalMem        " << (p.totalGlobalMem / 1000000) << " MB";
  KOUT(NON) << " BlockMem        " << (p.sharedMemPerBlock / 1000) << " KB";
  KOUT(NON) << " warpSize        " << p.warpSize;
  KOUT(NON) << " threadsPBlock   " << p.maxThreadsPerBlock;
  KOUT(NON) << " version:        " << p.runtime;
}

#endif /*MKN_GPU_ANY_INC_PRINFO_IPP*/
