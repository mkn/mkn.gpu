
#include <cassert>
#include <algorithm>

#include "mkn/gpu.hpp"
#include "mkn/gpu/asio.hpp"
#include "__share__.hpp"

static constexpr std::uint32_t BATCHES = 1;
static constexpr std::uint32_t NUM = 1024 * 1024 * BATCHES;
static constexpr std::uint32_t PER_BATCH = NUM / BATCHES;
static constexpr std::uint32_t TP_BLOCK = 1024;

struct A {
  std::uint32_t i0;
};

std::uint32_t test_single() {
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

  std::size_t checked = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j) {
      if (copy_back[j].i0 != a[j + offset].i0 + 1) return 1;
      ++checked;
    }
  }
  return checked != NUM;
}

struct B {
  float f0;
};

template <typename As>
std::uint32_t _test_multiple(As&& a) {
  for (std::uint32_t i = 0; i < NUM; ++i) a[i].i0 = i;
  std::vector<B> b(NUM / 1000);
  for (std::uint32_t i = 0; i < NUM / 1000; ++i) b[i].f0 = i + 1;

  auto batch = mkn::gpu::asio::Launcher{TP_BLOCK, BATCHES}(
      [] __device__(auto i, A* a, B* b) { a[i].i0 = a[i].i0 + b[i % 1000].f0; }, a, b);

  std::size_t checked = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j) {
      if (copy_back[j].i0 != a[j + offset].i0 + b[(j + offset) % 1000].f0) return 1;
      ++checked;
    }
  }
  return checked != NUM;
}

std::uint32_t test_multiple() { return _test_multiple(std::vector<A>(NUM)); }
std::uint32_t test_multiple_pinned() { return _test_multiple(mkn::gpu::HostArray<A, NUM>{}); }

template <typename Float = double>
std::uint32_t dev_class() {
  mkn::gpu::HostArray<Float, NUM> a;
  for (std::uint32_t i = 0; i < NUM; ++i) a[i] = i;

  std::vector<Float> b(NUM);
  for (std::uint32_t i = 0; i < NUM; ++i) b[i] = i + 1;
  DevClass<Float> dev(b);

  auto batch = mkn::gpu::asio::Launcher{TP_BLOCK, BATCHES}(
      [] __device__(auto i, auto* a, auto* b) { a[i] += (*b)[i]; }, a, dev);

  std::size_t checked = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j) {
      if (copy_back[j] != a[j + offset] + b[j + offset]) return 1;
      ++checked;
    }
  }
  return checked != NUM;
}

int main() {
  KOUT(NON) << __FILE__;

  return test_single() +           //
         test_multiple() +         //
         test_multiple_pinned() +  //
         dev_class<float>() +      //
         dev_class<double>();
}
