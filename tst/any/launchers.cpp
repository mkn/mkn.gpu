#include "mkn/gpu.hpp"

#include <stdexcept>
#include <type_traits>

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

std::uint32_t test_guess_span() {
  ManagedVector<float> mem(NUM, 1);
  auto* view = mem.data();

  DLauncher{mem}([=] __device__() { view[mkn::gpu::idx()] += 1; }).sync();
  GLauncher{mem}(kernel, mem, NUM).sync();

  for (auto const& e : mem)
    if (e != 3) return 1;
  return 0;
}

std::uint32_t test_stream() {
  ManagedVector<float> mem(NUM, 1);
  auto* view = mem.data();
  Stream stream;

  DLauncher{mem}.stream(stream, [=] __device__() { view[mkn::gpu::idx()] += 1; }).sync();

  for (auto const& e : mem)
    if (e != 2) return 1;
  return 0;
}

std::uint32_t test_empty() {
  ManagedVector<float> mem;
  DLauncher{mem}([] __device__() {}).sync();  // no launch
  return 0;
}

std::uint32_t test_fixed_rejects_indivisible() {
  try {
    GLauncher{NUM, 1, 16, 1};
  } catch (std::invalid_argument const&) {
    return 0;
  }
  return 1;
}

std::uint32_t test_guess_rejects_over_32_bit() {
  try {
    GLauncher{std::size_t{1} << 32};  // no allocation, the launcher only
  } catch (std::invalid_argument const&) {
    return 0;
  }
  return 1;
}

std::uint32_t test_clamp_bx_threads() {
  if (clamp_bx_threads(256, 1024, 32) != 256) return 1;  // under kernel max, as requested
  if (clamp_bx_threads(1024, 384, 32) != 384) return 1;  // register limited
  if (clamp_bx_threads(1024, 400, 32) != 384) return 1;  // warp multiple
  if (clamp_bx_threads(1024, 16, 32) != 16) return 1;    // below a warp
  return 0;
}

std::uint32_t test_guess_clamped() {
  ManagedVector<float> mem(NUM, 1);
  auto* view = mem.data();

  GLauncher glauncher{mem};
  glauncher.bx(kernel_bx_threads<bounded_kernel>())(bounded_kernel, mem, NUM).sync();
  DLauncher dlauncher{mem};
  dlauncher([=] __device__() { view[mkn::gpu::idx()] += 1; }).sync();

  for (auto const& e : mem)
    if (e != 3) return 1;
  if (glauncher.b.x > BOUNDED) return 1;
  if (glauncher.g.x * glauncher.b.x < NUM) return 1;
  KOUT(NON) << "bounded kernel block " << glauncher.b.x << " of cap " << glauncher.bx_cap;
  KOUT(NON) << "device lambda block " << dlauncher.b.x;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test_guess_span() +                 //
         test_stream() +                     //
         test_empty() +                      //
         test_fixed_rejects_indivisible() +  //
         test_guess_rejects_over_32_bit() +  //
         test_clamp_bx_threads() +           //
         test_guess_clamped();
}
