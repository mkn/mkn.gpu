// launches of a register heavy kernel, block size limited by the kernel not the device
//  guessed launches are clamped to the kernel limit, fixed launches use the block given, which
//  here is the kernel limit cached once by the caller in a local static
//  ncu --section LaunchStats --section Occupancy <test binary>

#include "mkn/gpu.hpp"
#include "mkn/kul/lang/cpp/compiler/def/unroll_loop.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>
#include <algorithm>
#include <ranges>

MKN_KUL_TEST_DECLARE();

using namespace mkn::gpu;

template <typename T>
using ManagedVector = std::vector<T, ManagedAllocator<T>>;

static constexpr std::uint32_t NUM = 1 << 20;
// grid covering size with blocks of bx, kernels are bounds checked
dim3 grid_for(std::size_t const size, std::size_t const bx) {
  return dim3(static_cast<std::uint32_t>(size / bx + (size % bx > 0)));
}

// every output the same in both and across them: every input is the same, so every output must be
//  too, an element a launch did not cover (e.g. a grid rounded down) keeps its initial 0
template <typename T>
void check_same(T const& a, T const& b) {
  MKN_KUL_TEST_CHECK(
      std::ranges::all_of(std::views::iota(std::size_t{0}, a.size()),
                          [&](auto const i) { return a[i] == b[i] && a[i] == a[0]; }),
      "a[i] == b[i] && a[i] == a[0]");
}

static constexpr std::uint32_t LIVE = 96;  // values held per thread, over 64 regs at 1024 threads

// all LIVE values stay in registers across the runtime trip count loop, which needs the inner
//  loops unrolled so every v[] index is a compile time constant, otherwise v is in local memory
__global__ void heavy(float* out, float const* in, std::size_t const size,
                      std::uint32_t const rounds) {
  auto const i = mkn::gpu::idx();
  if (i >= size) return;
  auto const x = in[i];
  float v[LIVE];
  MKN_KUL_UNROLL_LOOP(128)
  for (std::uint32_t k = 0; k < LIVE; ++k) v[k] = x + k;
  for (std::uint32_t r = 0; r < rounds; ++r) {
    MKN_KUL_UNROLL_LOOP(128)
    for (std::uint32_t k = 0; k < LIVE; ++k) v[k] = v[k] * x + v[(k + 1) % LIVE];
  }
  float sum = 0;
  MKN_KUL_UNROLL_LOOP(128)
  for (std::uint32_t k = 0; k < LIVE; ++k) sum += v[k];
  out[i] = sum;
}

void test_heavy() {
  ManagedVector<float> in(NUM, 0.5), out(NUM, 0), fixed_out(NUM, 0);

  static auto const bx = kernel_bx_threads<heavy>();
  GLauncher launcher{NUM};
  launcher.bx(bx)(heavy, out, in, NUM, 4u).sync();

  // the same kernel launched fixed with the cached limit as the block size
  GLauncher{grid_for(NUM, bx), dim3(static_cast<std::uint32_t>(bx))}(heavy, fixed_out, in, NUM, 4u)
      .sync();

  check_same(out, fixed_out);

  KOUT(NON) << "heavy kernel max block " << bx << ", device max " << getMaxThreadsPerBlock()
            << ", launched with " << launcher.b.x;
  MKN_KUL_TEST_CHECK(launcher.b.x <= bx);
}

void test_device_fit() {
  ManagedVector<float> in(NUM, 0.5), out(NUM, 0);

  auto heavy = [=] __device__(float* out, float const* in, std::size_t const size,
                              std::uint32_t const rounds) {
    auto const i = mkn::gpu::idx();
    if (i >= size) return;
    auto const x = in[i];
    float v[LIVE];
    MKN_KUL_UNROLL_LOOP(128)
    for (std::uint32_t k = 0; k < LIVE; ++k) v[k] = x + k;
    for (std::uint32_t r = 0; r < rounds; ++r) {
      MKN_KUL_UNROLL_LOOP(128)
      for (std::uint32_t k = 0; k < LIVE; ++k) v[k] = v[k] * x + v[(k + 1) % LIVE];
    }
    float sum = 0;
    MKN_KUL_UNROLL_LOOP(128)
    for (std::uint32_t k = 0; k < LIVE; ++k) sum += v[k];
    out[i] = sum;
  };

  DLauncher launcher{NUM};
  launcher(heavy, out, in, NUM, 4u).sync();

  // the same lambda launched fixed, the caller picks the block size, here the limit of the
  //  kernel a fixed launch generates (global_d_kernel), cached once in a local static
  ManagedVector<float> fixed_out(NUM, 0);
  static auto const fixed_bx = getDeviceKernelMaxThreadsPerBlock<Fixed>(heavy, out, in, NUM, 4u);
  DLauncher{grid_for(NUM, fixed_bx), dim3(static_cast<std::uint32_t>(fixed_bx))}(heavy, fixed_out,
                                                                                 in, NUM, 4u)
      .sync();

  // both launches must have covered every element
  check_same(out, fixed_out);

  auto const kernel_max = getDeviceKernelMaxThreadsPerBlock(heavy, out, in, NUM, 4u);
  KOUT(NON) << "heavy lambda max block " << kernel_max << " (fixed " << fixed_bx << "), device max "
            << getMaxThreadsPerBlock() << ", launched with " << launcher.b.x;
  MKN_KUL_TEST_CHECK(launcher.b.x <= kernel_max);
}

int main() {
  KOUT(NON) << __FILE__;
  test_heavy();
  test_device_fit();
}
