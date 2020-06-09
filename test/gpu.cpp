

#include "kul/gpu.hpp"

static constexpr size_t WIDTH = 1024, HEIGHT = 1024;
static constexpr size_t NUM = WIDTH * HEIGHT;
static constexpr size_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
__global__ void vectoradd(T* a, const T* b, const T* c) {
  int i = kul::gpu::idx();
  a[i] = b[i] + c[i];
}

int main() {
  kul::gpu::prinfo();
  std::vector<float> hostB(NUM), hostC(NUM);
  for (size_t i = 0; i < NUM; i++) hostB[i] = i;
  for (size_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  kul::gpu::DeviceMem<float> devA(NUM), devB(hostB), devC(hostC);
  kul::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<float>, devA.p, devB.p, devC.p);
  auto hostA = devA();
  for (size_t i = 0; i < NUM; i++)
    if (hostA[i] != (hostB[i] + hostC[i])) return 1;
  printf("PASSED!\n");
  return 0;
}
