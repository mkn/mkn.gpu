
#include "mkn/gpu.hpp"
#include "mkn/gpu/asio.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/span.hpp"
#include "mkn/kul/test.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <memory>

MKN_KUL_TEST_DECLARE();

static constexpr std::uint32_t BATCHES = 2;
static constexpr std::uint32_t NUM = 1024 * 1024 * BATCHES;
static constexpr std::uint32_t PER_BATCH = NUM / BATCHES;
static constexpr std::uint32_t TP_BLOCK = 1024;

struct A {
  std::uint32_t i0;
};

void test_single() {
  mkn::gpu::HostArray<A, NUM> a;
  for (std::uint32_t i = 0; i < NUM; ++i) a[i].i0 = i;

  auto batch = mkn::gpu::asio::Launcher{TP_BLOCK, BATCHES}(
      [] __device__(auto i, auto a) {
        assert(i < NUM);
        assert(a != nullptr);
        assert(a[i].i0 == i);
        a[i].i0 = i + 1;
      },
      a);

  std::size_t checked = 0, failed = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j) {
      failed += !(copy_back[j].i0 == a[j + offset].i0 + 1);
      ++checked;
    }
  }
  MKN_KUL_TEST_CHECK(failed == 0, "copy_back[j].i0 == a[j + offset].i0 + 1");
  MKN_KUL_TEST_CHECK(checked == NUM, "every element checked");
}

struct B {
  float f0;
};

template <typename As>
void _test_multiple(As&& a) {
  for (std::uint32_t i = 0; i < NUM; ++i) a[i].i0 = i;
  std::vector<B> b(NUM / 1000);
  for (std::uint32_t i = 0; i < NUM / 1000; ++i) b[i].f0 = i + 1;

  auto batch = mkn::gpu::asio::Launcher{TP_BLOCK, BATCHES}(
      [] __device__(auto i, A* a, B* b) { a[i].i0 = a[i].i0 + b[i % 1000].f0; }, a, b);

  std::size_t checked = 0, failed = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j) {
      failed += !(copy_back[j].i0 == a[j + offset].i0 + b[(j + offset) % 1000].f0);
      ++checked;
    }
  }
  MKN_KUL_TEST_CHECK(failed == 0,
                     "copy_back[j].i0 == a[j + offset].i0 + b[(j + offset) % 1000].f0");
  MKN_KUL_TEST_CHECK(checked == NUM, "every element checked");
}

void test_multiple() { _test_multiple(std::vector<A>(NUM)); }
void test_multiple_pinned() { _test_multiple(mkn::gpu::HostArray<A, NUM>{}); }

void test_lambda_copy_capture_views() {
  std::vector<A, mkn::gpu::ManagedAllocator<A>> a(NUM);
  for (std::uint32_t i = 0; i < NUM; ++i) a[i].i0 = i;
  std::vector<B, mkn::gpu::ManagedAllocator<B>> bv(NUM / 1000);
  auto b = bv.data();
  for (std::uint32_t i = 0; i < NUM / 1000; ++i) b[i].f0 = i + 1;

  auto batch = mkn::gpu::asio::Launcher{TP_BLOCK, BATCHES}(
      [=] __device__(auto i, A* a) { a[i].i0 = a[i].i0 + b[i % 1000].f0; }, a);

  std::size_t checked = 0, failed = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j) {
      failed += !(copy_back[j].i0 == a[j + offset].i0 + b[(j + offset) % 1000].f0);
      ++checked;
    }
  }
  MKN_KUL_TEST_CHECK(failed == 0,
                     "copy_back[j].i0 == a[j + offset].i0 + b[(j + offset) % 1000].f0");
  MKN_KUL_TEST_CHECK(checked == NUM, "every element checked");
}

int main() {
  KOUT(NON) << __FILE__;
  test_single();
  test_multiple();
  test_multiple_pinned();
  test_lambda_copy_capture_views();
}
