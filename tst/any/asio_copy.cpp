// mkn::gpu::asio::copy: copies queued on one stream behind a kernel on it, one stream sync for
//  all of them (no device sync), between managed and host memory; kernel -> a -> b -> back reads
//  each step's output, so a copy run out of order fails the check

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

  // a written by a kernel queued on the same stream as the copies: they run after it, and the
  //  one stream sync below is the only wait (no device sync)
  mkn::gpu::Stream stream;
  auto* pa = a.data();
  mkn::gpu::DLauncher launcher{NUM};
  launcher.s = stream();
  launcher([=] __device__() {
    auto const i = mkn::gpu::idx();
    pa[i] = 2.0 * i;
  });

  mkn::gpu::asio::copy(b, a, stream);                        // managed -> managed
  mkn::gpu::asio::copy(c.data(), host.data(), NUM, stream);  // host -> managed
  mkn::gpu::asio::copy(back.data(), b.data(), NUM,
                       stream);  // managed -> host, b as the first copy left it: in order
  mkn::gpu::asio::copy(b.data(), a.data(), 0, stream);  // nothing
  stream.sync();

  bool ok = true;
  for (std::uint32_t i = 0; i < NUM; ++i)
    ok = ok and b[i] == 2.0 * i and c[i] == T(i) and back[i] == 2.0 * i;
  MKN_KUL_TEST_CHECK(ok);
}

// no stream: a kernel and the device copy after it on the default stream, one device sync
void test_device_copy_default_stream() {
  using T = double;
  ManagedVector<T> a(NUM, 0), b(NUM, 0);
  auto* pa = a.data();
  mkn::gpu::DLauncher{NUM}([=] __device__() {  // no sync after the launch
    auto const i = mkn::gpu::idx();
    pa[i] = 3.0 * i;
  });
  mkn::gpu::copy_on_device_async(b.data(), a.data(), NUM);
  mkn::gpu::sync();

  bool ok = true;
  for (std::uint32_t i = 0; i < NUM; ++i) ok = ok and b[i] == 3.0 * i;
  MKN_KUL_TEST_CHECK(ok);
}

int main() {
  KOUT(NON) << __FILE__;
  test_copies_then_one_sync();
  test_device_copy_default_stream();
}
