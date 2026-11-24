
#include "mkn/gpu.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

MKN_KUL_TEST_DECLARE();

static constexpr uint32_t NUM = 5;

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

template <typename T>
using ManagedMemory = std::vector<T, mkn::gpu::NoConstructAllocator<T>>;

std::size_t alloced = 0;
struct S {
  S() { ++alloced; }

  std::uint16_t s = 1;
};

void test_does_construct_on_resize() {
  KLOG(INF);
  alloced = 0;
  ManagedVector<S> mem{NUM};
  mem.resize(NUM + NUM);
  MKN_KUL_TEST_CHECK(alloced == NUM + NUM, "resize constructs the new elements");
}

void test_does_not_construct_on_resize() {
  KLOG(INF);
  alloced = 0;
  ManagedMemory<S> mem{NUM};
  MKN_KUL_TEST_CHECK(mem.size() == 5);
  resize(mem, NUM + NUM);
  MKN_KUL_TEST_CHECK(mem.size() == 10);

  auto cap = mem.capacity();

  KLOG(INF) << mem.capacity();
  as_super(mem).emplace_back();  // does construct
  KLOG(INF) << mem.capacity();
  MKN_KUL_TEST_CHECK(mem.capacity() != cap);
  MKN_KUL_TEST_CHECK(mem[10].s == 1, "emplace_back constructs");

  cap = mem.capacity();

  resize(mem, NUM * 5);
  MKN_KUL_TEST_CHECK(mem.size() == 25);
  MKN_KUL_TEST_CHECK(mem.capacity() != cap);
  MKN_KUL_TEST_CHECK(mem[10].s == 1, "resize copies");

  KLOG(INF) << mem[10].s;
  KLOG(INF) << mem[11].s;
  MKN_KUL_TEST_CHECK(alloced == 1, "only emplace_back constructed");
  MKN_KUL_TEST_CHECK(mem[10].s == 1);
}

int main() {
  KOUT(NON) << __FILE__;
  test_does_construct_on_resize();
  test_does_not_construct_on_resize();
}
