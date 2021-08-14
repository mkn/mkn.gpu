
#include "kul/gpu.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

struct S {
  float f0 = 1;
  double d0 = 1;
};

__global__ void kernel(S* structs) {
  auto i = kul::gpu::idx();
  structs[i].f0 = structs[i].d0 + 1;
}

uint32_t test() {
  std::vector<S> host{NUM};
  for (uint32_t i = 0; i < NUM; ++i) host[i].d0 = i;
  kul::gpu::DeviceMem<S> dev{host};
  kul::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(kernel, dev);
  for (auto const& s : dev())
    if (s.f0 != s.d0 + 1) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test();
}
