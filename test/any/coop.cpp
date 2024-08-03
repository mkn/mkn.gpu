
#include "mkn/gpu.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

struct S {
  float f0 = 1;
  double d0 = 1;
};

std::uint32_t test_lambda_copy_capture_views() {
  mkn::gpu::GDLauncher<true, true> launcher{NUM};

  ManagedVector<S> mem{NUM};
  for (std::uint32_t i = 0; i < NUM; ++i) mem[i].d0 = i;

  auto* view = mem.data();

  launcher([=] __device__() {
    auto i = mkn::gpu::idx();
    mkn::gpu::grid_sync();
    view[i].f0 = view[i].d0 + 1;
  });

  for (std::uint32_t i = 0; i < NUM; ++i)
    if (view[i].f0 != view[i].d0 + 1) return 1;

  return 0;
}

int main() {
  KOUT(NON) << __FILE__;

  // if (mkn::gpu::supportsCooperativeLaunch()) return test_lambda_copy_capture_views();
  // KOUT(NON) << "Cooperative Launch not supported";
  return 0;
}
