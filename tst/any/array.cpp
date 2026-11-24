
#include "mkn/gpu.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <cstdint>
#include <algorithm>
#include <ranges>

MKN_KUL_TEST_DECLARE();

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
__global__ void vectoradd(T* a, T const* b, T const* c) {
  auto i = mkn::gpu::idx();
  a[i] = b[i] + c[i];
}

template <typename Float>
void test_1() {
  mkn::gpu::HostArray<Float, NUM> b, c;
  for (uint32_t i = 0; i < NUM; ++i) b[i] = i;
  for (uint32_t i = 0; i < NUM; ++i) c[i] = i * 100.0f;

  mkn::gpu::DeviceMem<Float> devA(NUM), devB(b), devC(c);
  mkn::gpu::GLauncher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(vectoradd<Float>,
                                                                               devA, devB, devC);

  auto a = devA();
  MKN_KUL_TEST_CHECK(std::ranges::all_of(std::views::iota(0u, NUM),
                                         [&](auto const i) { return a[i] == b[i] + c[i]; }),
                     "a[i] == b[i] + c[i]");
}

template <typename T>
__global__ void vectorinc(T* a) {
  auto i = mkn::gpu::idx();
  a[i] = a[i] + 1;
}

template <typename Float>
void test_2() {
  mkn::gpu::HostArray<Float, NUM> host;
  for (uint32_t i = 0; i < NUM; ++i) host[i] = i;

  mkn::gpu::DeviceMem<Float> dev{host};
  mkn::gpu::GLauncher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(vectorinc<Float>,
                                                                               dev);

  auto a = dev();
  MKN_KUL_TEST_CHECK(std::ranges::all_of(std::views::iota(0u, NUM),
                                         [&](auto const i) { return a[i] == host[i] + 1; }),
                     "a[i] == host[i] + 1");
}

int main() {
  KOUT(NON) << __FILE__;
  test_1<float>();
  test_1<double>();
  test_2<float>();
  test_2<double>();
}
