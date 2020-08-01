
#include "kul/gpu.hpp"

static constexpr size_t WIDTH = 1024, HEIGHT = 1024;
static constexpr size_t NUM = WIDTH * HEIGHT;
static constexpr size_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template<typename Float, bool GPU = false>
struct DevClass : kul::gpu::DeviceClass<GPU>
{
  using Super = kul::gpu::DeviceClass<GPU>;
  using gpu_t = DevClass<Float, true>;

  template<typename T>
  using container_t = typename Super::template container_t<T>;

  template<bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  DevClass(std::uint32_t nbr)
      : data{nbr}
  {
  }

  template<bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  DevClass(std::vector<Float> const& in)
      : data{in}
  {
  }

  template<bool gpu = GPU, std::enable_if_t<!gpu, bool> = 0>
  auto operator()()
  {
      return Super::template alloc<gpu_t>(data);
  }

  template<bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto& operator[](int i)  __device__  { return data[i]; }
  template<bool gpu = GPU, std::enable_if_t<gpu, bool> = 0>
  auto const& operator[](int i) const  __device__   { return data[i]; }

  container_t<Float> data;
};

template <typename T>
using GPUClass = typename ::DevClass<T>::gpu_t;

template <typename T>
__global__ void vectoradd(GPUClass<T>* a, GPUClass<T> const* b, GPUClass<T> const* c) {
  int i = kul::gpu::idx();
  (*a)[i] = (*b)[i] + (*c)[i];
}

template<typename Float>
size_t test(){
  std::vector<Float> hostB(NUM), hostC(NUM);
  for (size_t i = 0; i < NUM; i++) hostB[i] = i;
  for (size_t i = 0; i < NUM; i++) hostC[i] = i * 100.0f;
  DevClass<Float> devA(NUM), devB(hostB), devC(hostC);
  kul::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}(
      vectoradd<Float>, devA(), devB(), devC());
  auto hostA = devA.data();
  for (size_t i = 0; i < NUM; i++)
    if (hostA[i] != (hostB[i] + hostC[i])) return 1;
  return 0;
}

int main() {
  kul::gpu::prinfo();
  return test<float>() + test<double>();
}
