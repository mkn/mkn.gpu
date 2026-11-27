// launches with more dynamic shared memory than a kernel allows by default opt the kernel in
//  the default limit is for static plus dynamic shared memory, so a kernel with static shared
//  memory needs the opt-in for a dynamic request below the device default

#define MKN_GPU_FN_PER_NS 1
#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <ranges>

// MKN_GPU_FN_PER_NS: the backend header is used directly, not mkn/gpu.hpp
// IWYU pragma: no_include "mkn/gpu.hpp"
#include "mkn/gpu/cuda.hpp"  // IWYU pragma: keep
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

MKN_KUL_TEST_DECLARE();

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
void run(K kernel, std::size_t const dyn_bytes, float const expected) {
  auto const f = reinterpret_cast<void const*>(kernel);
  auto const before = getKernelMaxDynamicSharedMemory(f);

  DeviceMem<float> out(THREADS * BLOCKS);
  GLauncher launcher{dim3{BLOCKS}, dim3{THREADS}};
  launcher.ds = dyn_bytes;
  launcher(kernel, out, static_cast<std::uint32_t>(dyn_bytes / sizeof(float))).sync();

  KOUT(NON) << "dynamic " << dyn_bytes << " allowed before " << before << " after "
            << getKernelMaxDynamicSharedMemory(f);
  // the case under test needs the opt-in
  MKN_KUL_TEST_CHECK(dyn_bytes > before);
  MKN_KUL_TEST_CHECK(std::ranges::all_of(out(), [&](auto const& e) { return e == expected; }),
                     "e == expected");
}

// the opt-in maximum less the kernel's static shared memory launches; a byte more throws before
//  the launch, not as the attribute call's invalid value
void test_limit() {
  auto const f = reinterpret_cast<void const*>(with_static);
  std::size_t const limit = getMaxSharedMemoryPerBlockOptin(getDevice()) - STATIC * sizeof(float);
  run(with_static, limit, STATIC / THREADS * 1 + limit / 4 / THREADS * 2);

  DeviceMem<float> out(THREADS * BLOCKS);
  GLauncher launcher{dim3{BLOCKS}, dim3{THREADS}};
  launcher.ds = limit + 1;
  bool threw = false;
  try {
    launcher(with_static, out, std::uint32_t{0}).sync();
  } catch (std::runtime_error const& e) {
    threw = true;
    KOUT(NON) << e.what();
  }
  MKN_KUL_TEST_CHECK(threw, "over the opt-in maximum");
  MKN_KUL_TEST_CHECK(getKernelMaxDynamicSharedMemory(f) == limit);
}

int main() {
  KOUT(NON) << __FILE__;
  std::size_t constexpr KB = 1024;
  // 16KB static + 40KB dynamic, over the 48KB default although 40KB alone is not
  run(with_static, 40 * KB, STATIC / THREADS * 1 + 40 * KB / 4 / THREADS * 2);
  run(dynamic_only, 64 * KB, 64 * KB / 4 / THREADS * 2);
  test_limit();
}
