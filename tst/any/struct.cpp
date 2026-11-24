
#include "mkn/gpu.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <cstdint>
#include <vector>
#include <algorithm>
#include <ranges>

MKN_KUL_TEST_DECLARE();

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

struct S {
  float f0 = 1;
  double d0 = 1;
};

__global__ void kernel(S* structs) {
  auto i = mkn::gpu::idx();
  structs[i].f0 = structs[i].d0 + 1;
}

void test() {
  std::vector<S> host{NUM};
  for (uint32_t i = 0; i < NUM; ++i) host[i].d0 = i;
  mkn::gpu::DeviceMem<S> dev{host};
  mkn::gpu::GLauncher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(kernel, dev);
  MKN_KUL_TEST_CHECK(std::ranges::all_of(dev(), [&](auto const& s) { return s.f0 == s.d0 + 1; }),
                     "s.f0 == s.d0 + 1");
}

int main() {
  KOUT(NON) << __FILE__;
  test();
}
