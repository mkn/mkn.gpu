#define KUL_GPU_FN_PER_NS 1
#include "kul/gpu/rocm.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
__global__ void vectoradd(T* a, const T* b, const T* c) {
  int i = kul::gpu::hip::idx();
  a[i] = b[i] + c[i];
}

template<typename Float>
uint32_t test(){
  std::vector<Float> hostB(NUM), hostC(NUM);
  for (uint32_t i = 0; i < NUM; i++) hostB[i] = i;
  for (uint32_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  kul::gpu::hip::DeviceMem<Float> devA(NUM), devB(hostB), devC(hostC);
  kul::gpu::hip::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<Float>, devA, devB, devC);
  auto hostA = devA();
  for (uint32_t i = 0; i < NUM; i++)
    if (hostA[i] != (hostB[i] + hostC[i])) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test<float>() + test<double>();
}
