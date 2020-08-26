
#include "kul/gpu.hpp"

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template<typename Float>
struct A { Float* data; };

template<typename Float>
struct HostClassA : kul::gpu::HostClass {
  template <typename T>
  HostClassA(T t) : data{t} {}

  auto operator()() {
    return kul::gpu::HostClass::alloc<A<Float>>(data);
  }

  kul::gpu::DeviceMem<Float> data;
};

template<typename Float>
struct B { Float* data0, *data1; };

template<typename Float>
struct HostClassB : kul::gpu::HostClass {
  template <typename T>
  HostClassB(T t0, T t1) : data0{t0}, data1{t1} {}

  auto operator()() {
    return kul::gpu::HostClass::alloc<B<Float>>(data0, data1);
  }

  kul::gpu::DeviceMem<Float> data0, data1;
};

template <typename T>
__global__ void vectoradd(A<T> * a, B<T> const * const b) {
  auto i = kul::gpu::idx();
  a->data[i] = b->data0[i] + b->data1[i];
}

template<typename Float>
uint32_t test(){
  std::vector<Float> hostB(NUM), hostC(NUM);
  for (uint32_t i = 0; i < NUM; i++) hostB[i] = i;
  for (uint32_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  HostClassA<Float> devA(NUM);
  HostClassB<Float> devB{hostB, hostC};
  kul::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<Float>, devA(), devB());
  auto hostA = devA.data();
  for (uint32_t i = 0; i < NUM; i++)
    if (hostA[i] != (hostB[i] + hostC[i])) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test<float>() + test<double>();
}
