
#include "mkn/gpu.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
__global__ void vectoradd(T* a, T const* b, T const* c) {
  auto i = mkn::gpu::idx();
  a[i] = b[i] + c[i];
}

template <typename Float>
uint32_t test_1() {
  mkn::gpu::HostArray<Float, NUM> b, c;
  for (uint32_t i = 0; i < NUM; ++i) b[i] = i;
  for (uint32_t i = 0; i < NUM; ++i) c[i] = i * 100.0f;

  mkn::gpu::DeviceMem<Float> devA(NUM), devB(b), devC(c);
  mkn::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(vectoradd<Float>,
                                                                              devA, devB, devC);

  auto a = devA();
  for (uint32_t i = 0; i < NUM; ++i)
    if (a[i] != b[i] + c[i]) return 1;

  return 0;
}

template <typename T>
__global__ void vectorinc(T* a) {
  auto i = mkn::gpu::idx();
  a[i] = a[i] + 1;
}

template <typename Float>
uint32_t test_2() {
  mkn::gpu::HostArray<Float, NUM> host;
  for (uint32_t i = 0; i < NUM; ++i) host[i] = i;

  mkn::gpu::DeviceMem<Float> dev{host};
  mkn::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(vectorinc<Float>,
                                                                              dev);

  auto a = dev();
  for (uint32_t i = 0; i < NUM; ++i)
    if (a[i] != host[i] + 1) return 1;

  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test_1<float>() + test_1<double>() +  //
         test_2<float>() + test_2<double>();
}
