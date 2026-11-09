
#include "mkn/gpu.hpp"
#include "mkn/gpu/trace.hpp"
#include "mkn/kul/log.hpp"

#include <vector>
#include <cstddef>
#include <cstdint>

using namespace mkn::gpu;

template <typename T>
using ManagedVector = std::vector<T, ManagedAllocator<T>>;

static constexpr std::uint32_t NUM = 1000;

__global__ void kernel(float* data, std::size_t const size) {
  if (auto i = mkn::gpu::idx(); i < size) data[i] += 1;
}

std::uint32_t test_nested_ranges() {
  ManagedVector<float> v(NUM, 0);
  {
    trace::Range outer{"outer"};
    trace::mark("before launch");
    for (std::int64_t i = 0; i < 3; ++i) {
      trace::Range inner{"launch", i};
      GLauncher{NUM}(kernel, v.data(), NUM).sync();
    }
  }
  for (auto const& e : v)
    if (e != 3) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__ << " trace enabled: " << trace::enabled;
  return test_nested_ranges();
}
