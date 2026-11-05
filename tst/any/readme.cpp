// the README.md example, keep in sync

#include "mkn/gpu.hpp"

using namespace mkn::gpu;

static constexpr std::size_t N = 1'000'000;

std::uint32_t host_data() {
  std::vector<float> host(N, 1);  // plain host memory, the device cannot read it

  DeviceMem<float> dev{host};  // allocate on the device and copy host -> device
  DLauncher{N}([] __device__(float* d) { d[idx()] += 1; }, dev).sync();  // dev is passed as float*
  host = dev();  // copy device -> host

  for (auto const& e : host)
    if (e != 2) return 1;
  return 0;
}

std::uint32_t managed_data() {
  // managed memory, the same pointer is valid on host and device, pages migrate on access
  std::vector<float, ManagedAllocator<float>> managed(N, 1);

  DLauncher{managed}([] __device__(float* d) { d[idx()] += 1; }, managed).sync();  // no copies

  for (auto const& e : managed)
    if (e != 2) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return host_data() + managed_data();
}
