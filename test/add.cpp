
#include "kul/gpu.hpp"

static constexpr size_t WIDTH = 1024, HEIGHT = 1024;
static constexpr size_t NUM = WIDTH * HEIGHT;
static constexpr size_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
__global__ void vectoradd(T* a, const T* b, const T* c) {
  int i = kul::gpu::idx();
  a[i] = b[i] + c[i];
}

template<typename Float>
size_t test(){
  std::vector<Float> hostB(NUM), hostC(NUM);
  for (size_t i = 0; i < NUM; i++) hostB[i] = i;
  for (size_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  kul::gpu::DeviceMem<Float> devA(NUM), devB(hostB), devC(hostC);
  kul::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<Float>, devA, devB, devC);
  auto hostA = devA();
  for (size_t i = 0; i < NUM; i++)
    if (hostA[i] != (hostB[i] + hostC[i])) return 1;
  return 0;
}

int main() {
  kul::gpu::prinfo();
  return test<float>() + test<double>();
}
