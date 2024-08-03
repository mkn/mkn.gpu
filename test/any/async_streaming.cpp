
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <algorithm>

#include "mkn/kul/dbg.hpp"
#include "mkn/gpu/multi_launch.hpp"

using namespace std::chrono_literals;

std::uint32_t static constexpr NUM = 128 * 1024;  // ~ 1MB of doubles
std::size_t constexpr static C = 5;               // ~ 5MB of doubles

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

struct A {
  std::uint32_t i0;
};

std::uint32_t test() {
  using namespace mkn::gpu;
  using T = double;

  KUL_DBG_FUNC_ENTER;

  std::vector<ManagedVector<T>> vecs(C, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  StreamLauncher{vecs}
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 1; })
      .host([&](auto i) mutable {
        std::this_thread::sleep_for(200ms);
        for (auto& e : vecs[i]) e += 1;
      })
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 3; })();

  std::size_t val = 5;
  for (auto const& vec : vecs) {
    for (auto const& e : vec)
      if (e != val) return 1;
    ++val;
  };

  return 0;
}

std::uint32_t test_threaded(std::size_t const& nthreads = 2) {
  using namespace mkn::gpu;
  using T = double;

  KUL_DBG_FUNC_ENTER;

  std::vector<ManagedVector<T>> vecs(C, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  using namespace std::chrono_literals;

  ThreadedStreamLauncher{vecs, nthreads}
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 1; })
      .host([&](auto i) mutable {
        std::this_thread::sleep_for(200ms);
        for (auto& e : vecs[i]) e += 1;
      })
      .barrier()
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 3; })();

  std::size_t val = 5;
  for (auto const& vec : vecs) {
    for (auto const& e : vec)
      if (e != val) return 1;
    ++val;
  };

  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test() + test_threaded() + test_threaded(6);
}
