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

int main() {
  KOUT(NON) << __FILE__;
  return test_guess_span() +  //
         test_stream() +      //
         test_empty() +       //
         test_fixed_rejects_indivisible();
}
