
#include "mkn/gpu.hpp"
#include "mkn/kul/assert.hpp"

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

std::uint32_t test_does_construct_on_resize() {
  KLOG(INF);
  alloced = 0;
  ManagedVector<S> mem{NUM};
  mem.resize(NUM + NUM);
  return alloced != NUM + NUM;
}

std::uint32_t test_does_not_construct_on_resize() {
  KLOG(INF);
  alloced = 0;
  ManagedMemory<S> mem{NUM};
  mkn::kul::abort_if_not(mem.size() == 5 && "wrong size");
  resize(mem, NUM + NUM);
  mkn::kul::abort_if_not(mem.size() == 10 && "wrong size");

  auto cap = mem.capacity();

  KLOG(INF) << mem.capacity();
  as_super(mem).emplace_back();  // does construct
  KLOG(INF) << mem.capacity();
  mkn::kul::abort_if_not(mem.capacity() != cap && "capacity bad!");
  mkn::kul::abort_if_not(mem[10].s == 1 && "bad construct!");

  cap = mem.capacity();

  resize(mem, NUM * 5);
  mkn::kul::abort_if_not(mem.size() == 25 && "wrong size");
  mkn::kul::abort_if_not(mem.capacity() != cap && "capacity bad!");
  mkn::kul::abort_if_not(mem[10].s == 1 && "bad copy!");

  KLOG(INF) << mem[10].s;
  KLOG(INF) << mem[11].s;
  return alloced != 1 or mem[10].s != 1;
}

int main() {
  KOUT(NON) << __FILE__;

  return test_does_construct_on_resize() + test_does_not_construct_on_resize();
}
