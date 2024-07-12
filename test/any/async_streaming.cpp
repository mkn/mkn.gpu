
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <algorithm>

#include "mkn/gpu/multi_launch.hpp"

std::uint32_t static constexpr NUM = 128 * 1024 * 1024;  // ~ 1GB of doubles
std::size_t constexpr static C = 5;                      // ~ 5GB of doubles

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

struct A {
  std::uint32_t i0;
};

std::uint32_t test() {
  using namespace mkn::gpu;
  using T = double;
  std::vector<ManagedVector<T>> vecs(C, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  StreamLauncher{vecs}
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 1; })
      .host([&](auto i) mutable { vecs[i][0] += 1; })
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 3; })  //
      ();

  for (auto const& vec : vecs) std::cout << __LINE__ << " " << vec[0] << std::endl;

  return 0;
}

int main() {
  KOUT(NON) << __FILE__;

  return test();
}
