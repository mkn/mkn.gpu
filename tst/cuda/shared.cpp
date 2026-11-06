// launches with more dynamic shared memory than a kernel allows by default opt the kernel in
//  the default limit is for static plus dynamic shared memory, so a kernel with static shared
//  memory needs the opt-in for a dynamic request below the device default

#define MKN_GPU_FN_PER_NS 1
#include "mkn/gpu/cuda.hpp"

using namespace mkn::gpu::cuda;

static constexpr std::uint32_t THREADS = 256, BLOCKS = 64;
static constexpr std::uint32_t STATIC = 4096;  // floats, 16KB

// every thread writes and reads back across all of the shared memory, out is the per thread sum
__global__ void with_static(float* out, std::uint32_t const dyn_floats) {
  __shared__ float fixed[STATIC];
  extern __shared__ float dyn[];
  for (std::uint32_t j = threadIdx.x; j < STATIC; j += THREADS) fixed[j] = 1;
  for (std::uint32_t j = threadIdx.x; j < dyn_floats; j += THREADS) dyn[j] = 2;
  __syncthreads();
  float sum = 0;
  for (std::uint32_t j = threadIdx.x; j < STATIC; j += THREADS) sum += fixed[j];
  for (std::uint32_t j = threadIdx.x; j < dyn_floats; j += THREADS) sum += dyn[j];
  out[blockIdx.x * THREADS + threadIdx.x] = sum;
}

__global__ void dynamic_only(float* out, std::uint32_t const dyn_floats) {
  extern __shared__ float dyn[];
  for (std::uint32_t j = threadIdx.x; j < dyn_floats; j += THREADS) dyn[j] = 2;
  __syncthreads();
  float sum = 0;
  for (std::uint32_t j = threadIdx.x; j < dyn_floats; j += THREADS) sum += dyn[j];
  out[blockIdx.x * THREADS + threadIdx.x] = sum;
}

template <typename K>
std::uint32_t run(K kernel, std::size_t const dyn_bytes, float const expected) {
  auto const f = reinterpret_cast<void const*>(kernel);
  auto const before = getKernelMaxDynamicSharedMemory(f);

  DeviceMem<float> out(THREADS * BLOCKS);
  GLauncher launcher{dim3{BLOCKS}, dim3{THREADS}};
  launcher.ds = dyn_bytes;
  launcher(kernel, out, static_cast<std::uint32_t>(dyn_bytes / sizeof(float))).sync();

  KOUT(NON) << "dynamic " << dyn_bytes << " allowed before " << before << " after "
            << getKernelMaxDynamicSharedMemory(f);
  if (dyn_bytes <= before) return 1;  // the case under test needs the opt-in
  for (auto const& e : out())
    if (e != expected) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  std::size_t constexpr KB = 1024;
  // 16KB static + 40KB dynamic, over the 48KB default although 40KB alone is not
  return run(with_static, 40 * KB, STATIC / THREADS * 1 + 40 * KB / 4 / THREADS * 2)  //
         + run(dynamic_only, 64 * KB, 64 * KB / 4 / THREADS * 2);
}
