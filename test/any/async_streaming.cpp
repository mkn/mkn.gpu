
#include <thread>
#include <algorithm>

#include "mkn/kul/dbg.hpp"
#include "mkn/kul/time.hpp"
#include "mkn/gpu/multi_launch.hpp"

using namespace mkn::gpu;
using namespace std::chrono_literals;

std::uint32_t static constexpr NUM = 128 * 1024;  // ~ 1MB of doubles
std::size_t constexpr static C = 5;               // ~ 5MB of doubles

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

struct A {
  std::uint32_t i0;
};

std::uint32_t test() {
  KUL_DBG_FUNC_ENTER;
  using T = double;

  std::vector<ManagedVector<T>> vecs(C, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  auto const start = mkn::kul::Now::MILLIS();
  StreamLauncher{vecs}
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 1; })
      .host([&](auto i) mutable {
        std::this_thread::sleep_for(200ms);
        for (auto& e : vecs[i]) e += 1;
      })
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 3; })();
  auto const end = mkn::kul::Now::MILLIS();

  if (end - start > 1.5e3) return 1;

  std::size_t val = 5;
  for (auto const& vec : vecs) {
    for (auto const& e : vec)
      if (e != val) return 1;
    ++val;
  };

  return 0;
}

std::uint32_t test_threaded(std::size_t const& nthreads = 2) {
  KUL_DBG_FUNC_ENTER;
  using T = double;

  std::vector<ManagedVector<T>> vecs(C, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

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

std::uint32_t test_threaded_group_barrier(std::size_t const& nthreads = 2) {
  using T = double;
  KUL_DBG_FUNC_ENTER;

  std::vector<ManagedVector<T>> vecs(C + 1, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C + 1);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  auto const start = mkn::kul::Now::MILLIS();
  ThreadedStreamLauncher{vecs, nthreads}
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 1; })
      .host([&](auto i) mutable {
        std::this_thread::sleep_for(200ms);
        for (auto& e : vecs[i]) e += 1;
      })
      .group_barrier(3)
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 3; })();
  auto const end = mkn::kul::Now::MILLIS();

  if (end - start > 1e3) return 1;

  std::size_t val = 5;
  for (auto const& vec : vecs) {
    for (auto const& e : vec)
      if (e != val) return 1;
    ++val;
  };

  return 0;
}

std::uint32_t test_threaded_host_group_mutex(std::size_t const& nthreads = 2) {
  using T = double;
  KUL_DBG_FUNC_ENTER;

  std::size_t constexpr group_size = 3;
  std::vector<size_t> vals((C + 1) / group_size);  // 2 values;
  std::vector<ManagedVector<T>> vecs(C + 1, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C + 1);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  ThreadedStreamLauncher{vecs, nthreads}
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 1; })
      .host([&](auto i) mutable {
        std::this_thread::sleep_for(200ms);
        for (auto& e : vecs[i]) e += 1;
      })
      .host_group_mutex(group_size,  // lambda scope is locked per group
                        [&](auto const i) { vals[group_idx_modulo(group_size, i)] += i; })
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 3; })();

  if (vals != std::vector<size_t>{3, 12}) return 1;

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
  return test() + test_threaded() + test_threaded(6) + test_threaded_group_barrier() +
         test_threaded_host_group_mutex();
}
