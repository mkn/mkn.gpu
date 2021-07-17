
#include "kul/gpu/asio.hpp"

#include <algorithm>

static constexpr std::uint32_t BATCHES = 4;
static constexpr std::uint32_t NUM = 1024 * 1024 * BATCHES;
static constexpr std::uint32_t PER_BATCH = NUM / BATCHES;
static constexpr std::uint32_t TP_BLOCK = 256;

struct A {
  std::uint32_t i0;
};

__global__ void single(int offset, A* a) {
  auto i = kul::gpu::asio::idx() + offset;
  a[i].i0 = a[i].i0 + 1;
}

std::uint32_t test_single() {
  kul::gpu::HostArray<A, NUM> a;
  for (std::uint32_t i = 0; i < NUM; ++i) a[i].i0 = i;

  auto batch = kul::gpu::asio::Launcher{TP_BLOCK, BATCHES}(single, a);

  std::size_t err = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j)
      if (copy_back[j].i0 != a[j + offset].i0 + 1) return 1;
  }

  return 0;
}

struct B {
  float f0;
};

__global__ void multiple(int offset, A* a, B* b) {
  auto i = kul::gpu::asio::idx() + offset;
  a[i].i0 = a[i].i0 + b[i % 1000].f0;
}

std::uint32_t test_multiple() {
  std::vector<B> b{NUM / 1000};
  for (std::uint32_t i = 0; i < NUM / 1000; ++i) b[i].f0 = i + 1;
  std::vector<A> a(NUM);
  for (std::uint32_t i = 0; i < NUM; ++i) a[i].i0 = i;

  auto batch = kul::gpu::asio::Launcher{TP_BLOCK, BATCHES}(multiple, a, b);

  std::size_t err = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j)
      if (copy_back[j].i0 != a[j + offset].i0 + b[(j + offset) % 1000].f0) return 1;
  }
  return 0;
}

std::uint32_t test_multiple_pinned() {
  std::vector<B> b{NUM / 1000};
  for (std::uint32_t i = 0; i < NUM / 1000; ++i) b[i].f0 = i + 1;
  kul::gpu::HostArray<A, NUM> a;
  for (std::uint32_t i = 0; i < NUM; ++i) a[i].i0 = i;

  auto batch = kul::gpu::asio::Launcher{TP_BLOCK, BATCHES}(multiple, a, b);

  std::size_t err = 0;
  for (std::size_t i = 0; i < BATCHES; ++i) {
    auto offset = i * PER_BATCH;
    auto copy_back = batch->get(i);
    for (std::uint32_t j = 0; j < PER_BATCH; ++j)
      if (copy_back[j].i0 != a[j + offset].i0 + b[(j + offset) % 1000].f0) return 1;
  }
  return 0;
}

int main() {
  auto ret = test_single() + test_multiple() + test_multiple_pinned();
  KOUT(NON) << __FILE__ << " " << ret;
  return ret;
}
