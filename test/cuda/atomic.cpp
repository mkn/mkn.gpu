#define MKN_GPU_FN_PER_NS 1
#include "mkn/gpu/cuda.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

namespace NS0 {
template <typename T>
struct S {
  void operator+=(T const& v) __device__ { atomicAdd(&t, v); }
  void operator+=(T const&& v) __device__ { atomicAdd(&t, v); }

  T& t;
};
}  // namespace NS0

template <typename T>
__global__ void vectoradd(T* a, const T* b, const T* c) {
  auto i = mkn::gpu::cuda::idx();

  NS0::S<T>{a[i]} += b[i] + c[i];
  // atomicAdd(&a[i], );
}

template <typename Float>
uint32_t test() {
  std::vector<Float> hostB(NUM), hostC(NUM);
  for (uint32_t i = 0; i < NUM; i++) hostB[i] = i;
  for (uint32_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  mkn::gpu::cuda::DeviceMem<Float> devA(NUM), devB(hostB), devC(hostC);
  mkn::gpu::cuda::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<Float>, devA, devB, devC);
  auto hostA = devA();
  for (uint32_t i = 0; i < NUM; i++)
    if (hostA[i] != (hostB[i] + hostC[i])) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return 0;  // test<float>() + test<double>();
}
