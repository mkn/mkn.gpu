
#include "mkn/kul/log.hpp"
#include "mkn/gpu.hpp"
#include "mkn/gpu/def.hpp"
#include "mkn/kul/test.hpp"

#include <stdexcept>
#include <cstddef>
#include <initializer_list>
#include <vector>

MKN_KUL_TEST_DECLARE();

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

bool constexpr set_heap_limit = true;  // see if you need it by setting false after reboot

void test_allocate_percentage(double const percent) {
  auto const devProp = mkn::gpu::getDeviceProperties();

  auto const mem = devProp.totalGlobalMem;
  KLOG(DBG) << mem;

  auto const limit = mem * percent;
  if (set_heap_limit) mkn::gpu::setLimitMallocHeapSize(limit);

  auto const heapLim = mkn::gpu::getLimitMallocHeapSize();
  KLOG(DBG) << heapLim;
  if (set_heap_limit and !MKN_KUL_TEST_CHECK(limit <= heapLim, "the heap limit was set")) {
    KOUT(NON) << "Cannot set heap limit! " << limit << " " << heapLim;
    return;
  }

  if (limit < sizeof(double)) throw std::runtime_error("limit < sizeof(double)");

  auto const size = limit / sizeof(double);

  if (size == 0) throw std::runtime_error("size == 0");

  {
    ManagedVector<double> vec(size, 2);
    MKN_KUL_TEST_CHECK(vec.data()[vec.size() - 1] == 2);
  }

  KOUT(NON) << "Can allocate " << std::size_t(percent * 100)
            << "% of total mem: " << std::size_t(mem / 1e6) << "mb";
}

int main() {
  KOUT(NON) << __FILE__;
  mkn::gpu::prinfo();
  if constexpr (!mkn::gpu::CompileFlags::withCPU)  // NOT RELEVANT on the CPU
    for (double const percent : {.1, .2, .3, .4, .5}) test_allocate_percentage(percent);
}
