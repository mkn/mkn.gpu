// mkn::gpu::asio::copy: copies queued on one stream, one sync for all of them, between any of
//  managed, host (and device) memory

#include "mkn/gpu.hpp"
#include "mkn/gpu/asio.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

MKN_KUL_TEST_DECLARE();

static constexpr std::uint32_t NUM = 64 * 1024;

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

void test_copies_then_one_sync() {
  using T = double;
  ManagedVector<T> a(NUM, 0), b(NUM, 0), c(NUM, 0);
  std::vector<T> host(NUM, 0), back(NUM, 0);
  for (std::uint32_t i = 0; i < NUM; ++i) host[i] = i;

  // a written on the device first: the copy from it must see the kernel's values
  auto* pa = a.data();
  mkn::gpu::DLauncher{NUM}([=] __device__() {
    auto const i = mkn::gpu::idx();
    pa[i] = 2.0 * i;
  }).sync();

  mkn::gpu::Stream stream;
  mkn::gpu::asio::copy(b, a, stream);                        // managed -> managed
  mkn::gpu::asio::copy(c.data(), host.data(), NUM, stream);  // host -> managed
  mkn::gpu::asio::copy(back.data(), a.data(), NUM, stream);  // managed -> host
  mkn::gpu::asio::copy(b.data(), a.data(), 0, stream);       // nothing
  stream.sync();

  bool ok = true;
  for (std::uint32_t i = 0; i < NUM; ++i)
    ok = ok and b[i] == 2.0 * i and c[i] == T(i) and back[i] == 2.0 * i;
  MKN_KUL_TEST_CHECK(ok);
}

int main() {
  KOUT(NON) << __FILE__;
  test_copies_then_one_sync();
}
