
#include <stdexcept>
#include "mkn/gpu.hpp"

#include "mkn/gpu/def.hpp"
#include "mkn/kul/except.hpp"
#include "mkn/kul/log.hpp"

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

bool constexpr set_heap_limit = true;  // see if you need it by setting false after reboot

bool test_allocate_percentage(double const percent) {
  auto const devProp = mkn::gpu::getDeviceProperties();

  auto const mem = devProp.totalGlobalMem;
  KLOG(DBG) << mem;

  auto const limit = mem * percent;
  if (set_heap_limit) mkn::gpu::setLimitMallocHeapSize(limit);

  auto const heapLim = mkn::gpu::getLimitMallocHeapSize();
  KLOG(DBG) << heapLim;
  if (set_heap_limit and limit > heapLim) {
    KOUT(NON) << "Cannot set heap limit! " << limit << " " << heapLim;
    return 1;
  }

  if (limit < sizeof(double)) throw std::runtime_error("limit < sizeof(double)");

  auto const size = limit / sizeof(double);

  if (size == 0) throw std::runtime_error("size == 0");

  {
    ManagedVector<double> vec(size, 2);
    if (vec.data()[vec.size() - 1] != 2) return 1;
  }

  KOUT(NON) << "Can allocate " << std::size_t(percent * 100)
            << "% of total mem: " << std::size_t(mem / 1e6) << "mb";

  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  mkn::gpu::prinfo();
  if constexpr (mkn::gpu::CompileFlags::withCPU)
    return 0;  // NOT RELEVANT
  else
    return test_allocate_percentage(.1) +  //
           test_allocate_percentage(.2) +  //
           test_allocate_percentage(.3) +  //
           test_allocate_percentage(.4) +  //
           test_allocate_percentage(.5);
}
