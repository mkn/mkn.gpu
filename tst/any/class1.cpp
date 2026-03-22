
#include "mkn/gpu.hpp"
#include "__share__.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
using GPUClass = typename ::DevClass<T>::gpu_t;

template <typename T>
__global__ void vectoradd(GPUClass<T>* a, GPUClass<T> const* b, GPUClass<T> const* c) {
  auto i = mkn::gpu::idx();
  (*a)[i] = (*b)[i] + (*c)[i];
}

template <typename Float>
uint32_t test() {
  std::vector<Float> hostB(NUM), hostC(NUM);
  for (uint32_t i = 0; i < NUM; i++) hostB[i] = i;
  for (uint32_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  DevClass<Float> devA(NUM), devB(hostB), devC(hostC);
  mkn::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<Float>, devA(), devB(), devC());
  auto hostA = devA.data();
  for (uint32_t i = 0; i < NUM; i++)
    if (hostA[i] != (hostB[i] + hostC[i])) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test<float>() + test<double>();
}
