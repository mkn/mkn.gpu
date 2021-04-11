
#include "kul/gpu.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
__global__ void vectoradd(T* a, const T* b, const T* c) {
  auto i = kul::gpu::idx();
  a[i] = b[i] + c[i];
}

template<typename Float>
uint32_t test_1(){
  kul::gpu::HostArray<Float, NUM> hostB, hostC;
  for (uint32_t i = 0; i < NUM; ++i) hostB[i] = i;
  for (uint32_t i = 0; i < NUM; ++i) hostC[i] = i * 100.0f;

  kul::gpu::DeviceMem<Float> devA(NUM), devB(hostB), devC(hostC);
  kul::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<Float>, devA, devB, devC);

  auto hostA = devA();

  for (uint32_t i = 0; i < NUM; ++i)
    if (hostA[i] != hostB[i] + hostC[i]) return 1;

  return 0;
}

template <typename T>
__global__ void vectorinc(T* a) {
  auto i = kul::gpu::idx();
  a[i] = a[i] + 1;
}

template<typename Float>
uint32_t test_2(){
  kul::gpu::HostArray<Float, NUM> host;
  for (uint32_t i = 0; i < NUM; ++i) host[i] = i;

  kul::gpu::DeviceMem<Float> dev{host};
  kul::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectorinc<Float>, dev);

  auto hostA = dev();

  for (uint32_t i = 0; i < NUM; ++i)
    if (hostA[i] != host[i] + 1) return 1;

  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test_1<float>() + test_1<double>() + test_2<float>() + test_2<double>();
}
