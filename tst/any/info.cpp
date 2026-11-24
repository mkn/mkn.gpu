
#include "mkn/gpu.hpp"
#include "mkn/gpu/def.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <cstdint>
#include <vector>

MKN_KUL_TEST_DECLARE();

// the warp size mkn.gpu reports is the one kernels launched through it see
void test_warp_size() {
  auto const& props = mkn::gpu::getDeviceProperties();
  MKN_KUL_TEST_CHECK(props.warpSize > 0);
  MKN_KUL_TEST_CHECK(props.warpSize == mkn::gpu::warp_size());
#if !MKN_GPU_CPU  // no warpSize in host code
  std::vector<std::uint32_t, mkn::gpu::ManagedAllocator<std::uint32_t>> seen(1, 0);
  auto* view = seen.data();
  mkn::gpu::DLauncher{1}([=] __device__() { view[0] = warpSize; }).sync();
  MKN_KUL_TEST_CHECK(seen[0] == props.warpSize, "a kernel's warpSize is the reported one");
#endif
}

int main() {
  KOUT(NON) << __FILE__;
  mkn::gpu::prinfo();
  test_warp_size();
}
