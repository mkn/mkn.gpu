
#include "mkn/gpu.hpp"

#include "mkn/kul/assert.hpp"

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
std::uint32_t _test(L&& launcher) {
  ManagedVector<S> mem{NUM};
  if constexpr (!mkn::gpu::CompileFlags::withCPU)
    mkn::kul::abort_if_not(mkn::gpu::Pointer{mem.data()}.is_managed_ptr() && "not host pointer");

  for (std::uint32_t i = 0; i < NUM; ++i) mem[i].d0 = i;

  launcher(kernel, mem);

  for (std::uint32_t i = 0; i < NUM; ++i)
    if (mem[i].f0 != mem[i].d0 + 1) return 1;

  return 0;
}

std::uint32_t test() {
  return _test(mkn::gpu::Launcher{WIDTH, HEIGHT, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y});
}

std::uint32_t test_guess() { return _test(mkn::gpu::GLauncher{NUM}); }

template <typename L>
std::uint32_t _test_lambda_copy_capture_views(L&& launcher) {
  ManagedVector<S> mem{NUM};
  for (std::uint32_t i = 0; i < NUM; ++i) mem[i].d0 = i;

  auto* view = mem.data();
  launcher([=] __device__() {
    auto i = mkn::gpu::idx();
    view[i].f0 = view[i].d0 + 1;
  });

  for (std::uint32_t i = 0; i < NUM; ++i)
    if (view[i].f0 != view[i].d0 + 1) return 1;

  return 0;
}

std::uint32_t test_lambda_copy_capture_views() {
  return _test_lambda_copy_capture_views(mkn::gpu::GDLauncher{NUM});
}

std::uint32_t test_lambda_ref_copy_capture_views() {
  mkn::gpu::GDLauncher launcher{NUM};

  ManagedVector<S> mem{NUM};
  for (std::uint32_t i = 0; i < NUM; ++i) mem[i].d0 = i;

  auto* view = mem.data();

  auto fn = [=] __device__() {
    auto i = mkn::gpu::idx();
    view[i].f0 = view[i].d0 + 1;
  };

  launcher(fn);

  for (std::uint32_t i = 0; i < NUM; ++i)
    if (view[i].f0 != view[i].d0 + 1) return 1;

  return 0;
}

std::uint32_t test_zero() {
  auto const size = 1000;  // not warp size divisible!
  ManagedVector<float> mem0(size, 1);
  ManagedVector<float> mem1(size, 2);

  auto* view0 = mem0.data();
  auto* view1 = mem1.data();

  mkn::gpu::DLauncher()([=] __device__() {
    mkn::gpu::fill_warp_size(view0, size, 0.0f);
    mkn::gpu::fill_warp_size(view1, size, 0.0f);
  });

  for (std::uint32_t i = 0; i < size; ++i)
    if (mem0[i] + mem1[i] != 0) return 1;

  return 0;
}

uint32_t test_copy() {
  std::vector<float> hst0(NUM, 1), hst1(NUM, 2);
  ManagedVector<float> dev0(NUM), dev1(NUM);

  // copy(T0* dst, T1* src, Size const size)
  mkn::gpu::copy(dev0, hst0);
  if (dev0.back() != 1) return 1;
  mkn::gpu::copy(hst1, hst0);
  if (hst1.back() != 1) return 1;
  mkn::gpu::copy(dev1, hst1);

  return dev1[NUM - 1] != 1;
}

int main() {
  KOUT(NON) << __FILE__;
  return test() + test_zero() +                  //
         test_guess() +                          //
         test_lambda_copy_capture_views() +      //
         test_lambda_ref_copy_capture_views() +  //
         test_copy();
}
