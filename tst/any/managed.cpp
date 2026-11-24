
#include "mkn/gpu.hpp"
#include "mkn/gpu/def.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>
#include <ranges>

MKN_KUL_TEST_DECLARE();

static constexpr uint32_t WIDTH = 1024, HEIGHT = 1024;
static constexpr uint32_t NUM = WIDTH * HEIGHT;
static constexpr uint32_t THREADS_PER_BLOCK_X = 16, THREADS_PER_BLOCK_Y = 16;

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

struct S {
  float f0 = 1;
  double d0 = 1;
};

__global__ void kernel(S* structs) {
  auto i = mkn::gpu::idx();
  structs[i].f0 = structs[i].d0 + 1;
}

template <typename L>
void _test(L&& launcher) {
  ManagedVector<S> mem{NUM};
  if constexpr (!mkn::gpu::CompileFlags::withCPU)
    MKN_KUL_TEST_CHECK(mkn::gpu::Pointer{mem.data()}.is_managed_ptr());

  for (std::uint32_t i = 0; i < NUM; ++i) mem[i].d0 = i;

  launcher(kernel, mem).sync();

  MKN_KUL_TEST_CHECK(std::ranges::all_of(std::views::iota(0u, NUM),
                                         [&](auto const i) { return mem[i].f0 == mem[i].d0 + 1; }),
                     "mem[i].f0 == mem[i].d0 + 1");
}

void test() { _test(mkn::gpu::GLauncher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y}); }

void test_guess() { _test(mkn::gpu::GLauncher{NUM}); }

template <typename L>
void _test_lambda_copy_capture_views(L&& launcher) {
  ManagedVector<S> mem{NUM};
  for (std::uint32_t i = 0; i < NUM; ++i) mem[i].d0 = i;

  auto* view = mem.data();
  launcher([=] __device__() {
    auto i = mkn::gpu::idx();
    view[i].f0 = view[i].d0 + 1;
  }).sync();

  MKN_KUL_TEST_CHECK(
      std::ranges::all_of(std::views::iota(0u, NUM),
                          [&](auto const i) { return view[i].f0 == view[i].d0 + 1; }),
      "view[i].f0 == view[i].d0 + 1");
}

void test_lambda_copy_capture_views() { _test_lambda_copy_capture_views(mkn::gpu::DLauncher{NUM}); }

void test_lambda_ref_copy_capture_views() {
  mkn::gpu::DLauncher launcher{NUM};

  ManagedVector<S> mem{NUM};
  for (std::uint32_t i = 0; i < NUM; ++i) mem[i].d0 = i;

  auto* view = mem.data();

  auto fn = [=] __device__() {
    auto i = mkn::gpu::idx();
    view[i].f0 = view[i].d0 + 1;
  };

  launcher(fn).sync();

  MKN_KUL_TEST_CHECK(
      std::ranges::all_of(std::views::iota(0u, NUM),
                          [&](auto const i) { return view[i].f0 == view[i].d0 + 1; }),
      "view[i].f0 == view[i].d0 + 1");
}

void test_zero() {
  auto const size = 1000;  // not warp size divisible!
  ManagedVector<float> mem0(size, 1);
  ManagedVector<float> mem1(size, 2);

  auto* view0 = mem0.data();
  auto* view1 = mem1.data();

  mkn::gpu::DLauncher()([=] __device__() {
    mkn::gpu::fill_warp_size(view0, size, 0.0f);
    mkn::gpu::fill_warp_size(view1, size, 0.0f);
  }).sync();

  MKN_KUL_TEST_CHECK(std::ranges::all_of(std::views::iota(0, size),
                                         [&](auto const i) { return mem0[i] + mem1[i] == 0; }),
                     "mem0[i] + mem1[i] == 0");
}

void test_copy() {
  std::vector<float> hst0(NUM, 1), hst1(NUM, 2);
  ManagedVector<float> dev0(NUM), dev1(NUM);

  // copy(T0* dst, T1* src, Size const size)
  mkn::gpu::copy(dev0, hst0);
  MKN_KUL_TEST_CHECK(dev0.back() == 1);
  mkn::gpu::copy(hst1, hst0);
  MKN_KUL_TEST_CHECK(hst1.back() == 1);
  mkn::gpu::copy(dev1, hst1);

  MKN_KUL_TEST_CHECK(dev1[NUM - 1] == 1);
}

int main() {
  KOUT(NON) << __FILE__;
  test();
  test_zero();
  test_guess();
  test_lambda_copy_capture_views();
  test_lambda_ref_copy_capture_views();
  test_copy();
}
