
#include "mkn/gpu.hpp"

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
  if constexpr (!mkn::gpu::CompileFlags::withCPU) {
    assert(mkn::gpu::Pointer{mem.data()}.is_managed_ptr());
  }

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

int main() {
  KOUT(NON) << __FILE__;
  return test() +                            //
         test_guess() +                      //
         test_lambda_copy_capture_views() +  //
         test_lambda_ref_copy_capture_views();
}
