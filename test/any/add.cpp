
#include "mkn/kul/gpu.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t TPB_X = 16, TPB_Y = 16;

template <typename T>
__global__ void vector_inc0(T* a) {
  auto i = mkn::gpu::idx();
  a[i] = i + 1;
}

template <typename Float>
uint32_t test_inc() {
  mkn::gpu::DeviceMem<Float> devA(NUM);
  mkn::gpu::Launcher{WIDTH, HEIGHT, TPB_X, TPB_Y}(vector_inc0<Float>, devA);
  auto a = devA();
  for (uint32_t i = 0; i < NUM; i++)
    if (a[i] != i + 1) return 1;
  return 0;
}

template <typename T>
__global__ void vectoradd1(T* a, T* b) {
  auto i = mkn::gpu::idx();
  a[i] = b[i] + 1;
}

template <typename Float>
uint32_t test_add1() {
  std::vector<Float> b(NUM);
  for (uint32_t i = 0; i < NUM; i++) b[i] = i;
  mkn::gpu::DeviceMem<Float> devA(NUM), devB(b);
  mkn::gpu::Launcher{WIDTH, HEIGHT, TPB_X, TPB_Y}(vectoradd1<Float>, devA, devB);
  auto a = devA();
  for (uint32_t i = 0; i < NUM; i++)
    if (a[i] != b[i] + 1) return 1;
  return 0;
}

template <typename T>
__global__ void vectoradd2(T* a, T const* const b, T const* const c) {
  auto i = mkn::gpu::idx();
  a[i] = b[i] + c[i];
}
template <typename Float>
uint32_t test_add2() {
  std::vector<Float> b(NUM), c(NUM);
  for (uint32_t i = 0; i < NUM; i++) b[i] = i;
  for (uint32_t i = 0; i < NUM; i++) c[i] = i * 100.0f;
  mkn::gpu::DeviceMem<Float> devA(NUM), devB(b), devC(c);
  mkn::gpu::Launcher{WIDTH, HEIGHT, TPB_X, TPB_Y}(vectoradd2<Float>, devA, devB, devC);
  auto a = devA();
  for (uint32_t i = 0; i < NUM; i++)
    if (a[i] != b[i] + c[i]) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test_inc<float>() + test_inc<double>() +    //
         test_add1<float>() + test_add1<double>() +  //
         test_add2<float>() + test_add2<double>();
}
