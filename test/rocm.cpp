

#include "kul/gpu/rocm.hpp"

static constexpr size_t WIDTH = 1024, HEIGHT = 1024;
static constexpr size_t NUM = WIDTH * HEIGHT;
static constexpr size_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

int main() {
  kul::gpu::prinfo();
  std::vector<float> hostB(NUM), hostC(NUM);
  for (size_t i = 0; i < NUM; i++) hostB[i] = i;
  for (size_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  kul::gpu::DeviceMem<float> devA(NUM), devB(hostB), devC(hostC);
  kul::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      kul::gpu::vectoradd<float>, devA.p, devB.p, devC.p, WIDTH, HEIGHT);
  auto hostA = devA.get();
  for (size_t i = 0; i < NUM; i++)
    if (hostA[i] != (hostB[i] + hostC[i])) return 1;
  printf("PASSED!\n");
  return 0;
}

