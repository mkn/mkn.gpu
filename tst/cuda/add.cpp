#define MKN_GPU_FN_PER_NS 1
#include <cstdint>
#include <vector>
#include <algorithm>
#include <ranges>

// MKN_GPU_FN_PER_NS: the backend header is used directly, not mkn/gpu.hpp
// IWYU pragma: no_include "mkn/gpu.hpp"
#include "mkn/gpu/cuda.hpp"  // IWYU pragma: keep
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

MKN_KUL_TEST_DECLARE();

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
__global__ void vectoradd(T* a, T const* b, T const* c) {
  auto i = mkn::gpu::cuda::idx();
  a[i] = b[i] + c[i];
}

template <typename Float>
void test() {
  std::vector<Float> hostB(NUM), hostC(NUM);
  for (uint32_t i = 0; i < NUM; i++) hostB[i] = i;
  for (uint32_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  mkn::gpu::cuda::DeviceMem<Float> devA(NUM), devB(hostB), devC(hostC);
  mkn::gpu::cuda::GLauncher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<Float>, devA, devB, devC);
  auto hostA = devA();
  MKN_KUL_TEST_CHECK(
      std::ranges::all_of(std::views::iota(0u, NUM),
                          [&](auto const i) { return hostA[i] == (hostB[i] + hostC[i]); }),
      "hostA[i] == (hostB[i] + hostC[i])");
}

int main() {
  KOUT(NON) << __FILE__;
  test<float>();
  test<double>();
}
