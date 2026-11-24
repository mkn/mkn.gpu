#include "mkn/gpu.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>
#include <ranges>

MKN_KUL_TEST_DECLARE();

using namespace mkn::gpu;

template <typename T>
using ManagedVector = std::vector<T, ManagedAllocator<T>>;

static constexpr std::uint32_t NUM = 1000;  // not block size divisible

__global__ void kernel(float* data, std::size_t const size) {
  if (auto i = mkn::gpu::idx(); i < size) data[i] += 1;
}

// kernel limit below any device maximum, guessed launches must not exceed it
static constexpr std::uint32_t BOUNDED = 128;

__global__ void __launch_bounds__(BOUNDED) bounded_kernel(float* data, std::size_t const size) {
  if (auto i = mkn::gpu::idx(); i < size) data[i] += 1;
}

static_assert(std::is_same_v<decltype(GlobalLauncher{NUM}), GlobalLauncher<Guess>>);
static_assert(std::is_same_v<decltype(DeviceLauncher{NUM}), DeviceLauncher<Guess>>);
static_assert(std::is_same_v<decltype(GLauncher{NUM}), GLauncher<Guess>>);
static_assert(std::is_same_v<decltype(DLauncher{NUM}), DLauncher<Guess>>);
static_assert(std::is_same_v<decltype(DLauncher{NUM, 0}), DLauncher<Guess>>);
static_assert(std::is_same_v<decltype(DLauncher{ManagedVector<float>{}}), DLauncher<Guess>>);
static_assert(std::is_same_v<decltype(DLauncher{}), DLauncher<Fixed>>);
static_assert(std::is_same_v<decltype(DLauncher{dim3{1}, dim3{32}}), DLauncher<Fixed>>);
static_assert(std::is_same_v<decltype(GLauncher{64, 64, 16, 16}), GLauncher<Fixed>>);

void test_guess_span() {
  ManagedVector<float> mem(NUM, 1);
  auto* view = mem.data();

  DLauncher{mem}([=] __device__() { view[mkn::gpu::idx()] += 1; }).sync();
  GLauncher{mem}(kernel, mem, NUM).sync();

  MKN_KUL_TEST_CHECK(std::ranges::all_of(mem, [&](auto const& e) { return e == 3; }), "e == 3");
}

void test_stream() {
  ManagedVector<float> mem(NUM, 1);
  auto* view = mem.data();
  Stream stream;

  DLauncher{mem}.stream(stream, [=] __device__() { view[mkn::gpu::idx()] += 1; }).sync();

  MKN_KUL_TEST_CHECK(std::ranges::all_of(mem, [&](auto const& e) { return e == 2; }), "e == 2");
}

void test_empty() {
  ManagedVector<float> mem;
  DLauncher{mem}([] __device__() {}).sync();  // no launch
}

template <typename F>
bool throws(F&& f) {
  try {
    f();
  } catch (std::invalid_argument const&) {
    return true;
  }
  return false;
}

void test_fixed_rejects_indivisible() {
  MKN_KUL_TEST_CHECK(throws([] { GLauncher{NUM, 1, 16, 1}; }), "a grid its block does not divide");
}

// throws on launch, not construction, so a smaller block from .bx() is checked instead
void test_guess_rejects_over_32_bit() {
  GLauncher launcher{std::size_t{1} << 32};
  // throws before launching
  MKN_KUL_TEST_CHECK(
      throws([&] { launcher(kernel, static_cast<float*>(nullptr), std::size_t{0}); }),
      "a grid over 32 bits");
}

void test_guess_grid_dim() {
  std::size_t constexpr max = std::numeric_limits<std::uint32_t>::max();
  MKN_KUL_TEST_CHECK(guess_grid_dim(1000, 256) == 4);
  // fits with 128
  MKN_KUL_TEST_CHECK(guess_grid_dim(max - 511, 128) == (max - 511) / 128);
  // not with 1024
  MKN_KUL_TEST_CHECK(throws([] { guess_grid_dim(max - 511, 1024); }));
  // no wrap
  MKN_KUL_TEST_CHECK(throws([] { guess_grid_dim(std::numeric_limits<std::size_t>::max(), 1024); }));
  // no division by 0
  MKN_KUL_TEST_CHECK(throws([] { guess_grid_dim(1, 0); }));
}

void test_clamp_bx_threads() {
  // under kernel max, as requested
  MKN_KUL_TEST_CHECK(clamp_bx_threads(256, 1024, 32) == 256);
  // register limited
  MKN_KUL_TEST_CHECK(clamp_bx_threads(1024, 384, 32) == 384);
  // warp multiple
  MKN_KUL_TEST_CHECK(clamp_bx_threads(1024, 400, 32) == 384);
  // below a warp
  MKN_KUL_TEST_CHECK(clamp_bx_threads(1024, 16, 32) == 16);
}

void test_guess_clamped() {
  ManagedVector<float> mem(NUM, 1);
  auto* view = mem.data();

  GLauncher glauncher{mem};
  glauncher.bx(kernel_bx_threads<bounded_kernel>())(bounded_kernel, mem, NUM).sync();
  DLauncher dlauncher{mem};
  dlauncher([=] __device__() { view[mkn::gpu::idx()] += 1; }).sync();

  MKN_KUL_TEST_CHECK(std::ranges::all_of(mem, [&](auto const& e) { return e == 3; }), "e == 3");
  MKN_KUL_TEST_CHECK(glauncher.b.x <= BOUNDED);
  MKN_KUL_TEST_CHECK(glauncher.g.x * glauncher.b.x >= NUM);
  KOUT(NON) << "bounded kernel block " << glauncher.b.x << " of cap " << glauncher.bx_cap;
  KOUT(NON) << "device lambda block " << dlauncher.b.x;
}

int main() {
  KOUT(NON) << __FILE__;
  test_guess_span();
  test_stream();
  test_empty();
  test_fixed_rejects_indivisible();
  test_guess_rejects_over_32_bit();
  test_guess_grid_dim();
  test_clamp_bx_threads();
  test_guess_clamped();
}
