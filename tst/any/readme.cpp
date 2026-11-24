// the README.md example, keep in sync

#include "mkn/gpu.hpp"
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

static constexpr std::size_t N = 1'000'000;

void host_data() {
  std::vector<float> host(N, 1);  // plain host memory, the device cannot read it

  DeviceMem<float> dev{host};  // allocate on the device and copy host -> device
  DLauncher{N}([] __device__(float* d) { d[idx()] += 1; }, dev).sync();  // dev is passed as float*
  host = dev();                                                          // copy device -> host

  MKN_KUL_TEST_CHECK(std::ranges::all_of(host, [&](auto const& e) { return e == 2; }), "e == 2");
}

void managed_data() {
  // managed memory, the same pointer is valid on host and device, pages migrate on access
  std::vector<float, ManagedAllocator<float>> managed(N, 1);

  DLauncher{managed}([] __device__(float* d) { d[idx()] += 1; }, managed).sync();  // no copies

  MKN_KUL_TEST_CHECK(std::ranges::all_of(managed, [&](auto const& e) { return e == 2; }), "e == 2");
}

int main() {
  KOUT(NON) << __FILE__;
  host_data();
  managed_data();
}
