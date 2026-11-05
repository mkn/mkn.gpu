

#include "mkn/gpu.hpp"
#include "mkn/kul/dbg.hpp"
#include "mkn/kul/time.hpp"
#include "mkn/gpu/stream.hpp"

#include <algorithm>

using namespace mkn::gpu;
using namespace std::chrono_literals;

std::uint32_t static constexpr NUM = 128 * 1024;  // ~ 1MB of doubles
std::size_t constexpr static C = 5;               // ~ 5MB of doubles

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

std::uint32_t test() {
  MKN_KUL_DBG_FUNC_ENTER;
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

  if (end - start > 1.5e3) return 1;  // serial, C host steps of 200ms, ~1s, nothing else adds up

  std::size_t val = 5;  // item i starts at i, + 1 (dev) + 1 (host) + 3 (dev)
  for (auto const& vec : vecs) {
    for (auto const& e : vec)
      if (e != val) return 1;
    ++val;
  };

  return 0;
}

std::uint32_t test_threaded(std::size_t const& nthreads = 2) {
  MKN_KUL_DBG_FUNC_ENTER;
  using T = double;

  std::vector<ManagedVector<T>> vecs(C, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  ThreadedStreamLauncher tsl{vecs, nthreads};
  tsl.dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 1; })
      .host([&](auto i) mutable {
        std::this_thread::sleep_for(200ms);
        for (auto& e : vecs[i]) e += 1;
      })
      .barrier()
      .dev([=] __device__(auto i) { views[i][mkn::gpu::idx()] += 3; })();

  std::size_t val = 5;  // item i starts at i, + 1 (dev) + 1 (host) + 3 (dev)
  for (auto const& vec : vecs) {
    for (auto const& e : vec)
      if (e != val) return 1;
    ++val;
  };

  tsl.print_times();

  return 0;
}

std::uint32_t test_threaded_group_barrier(std::size_t const& nthreads = 2) {
  using T = double;
  MKN_KUL_DBG_FUNC_ENTER;

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
      .group_barrier(3)  // items continue once all 3 items of their group arrive
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 3; })();
  auto const end = mkn::kul::Now::MILLIS();

  if (end - start > 1e3) return 1;  // C + 1 host steps of 200ms over 3 threads, serial is 1.2s

  std::size_t val = 5;  // item i starts at i, + 1 (dev) + 1 (host) + 3 (dev)
  for (auto const& vec : vecs) {
    for (auto const& e : vec)
      if (e != val) return 1;
    ++val;
  };

  return 0;
}

std::uint32_t test_threaded_host_group_mutex(std::size_t const& nthreads = 2) {
  using T = double;
  MKN_KUL_DBG_FUNC_ENTER;

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

  // once per item, summed per group of 3: 0 + 1 + 2 and 3 + 4 + 5
  if (vals != std::vector<size_t>{3, 12}) return 1;

  std::size_t val = 5;  // item i starts at i, + 1 (dev) + 1 (host) + 3 (dev)
  for (auto const& vec : vecs) {
    for (auto const& e : vec)
      if (e != val) return 1;
    ++val;
  };

  return 0;
}

std::uint32_t test_threaded_host_group_idx(std::size_t const& nthreads = 2) {
  using T = double;
  MKN_KUL_DBG_FUNC_ENTER;

  std::size_t constexpr group_size = 3;

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
      .host_group_idx(group_size, 0,  // only items i % group_size == 0, + 1 more
                      [&](auto const i) {
                        for (auto& e : vecs[i]) e += 1;
                      })
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 3; })();

  std::size_t val = 5;  // item i starts at i, + 1 (dev) + 1 (host) + 3 (dev)
  for (std::size_t i = 0; i < vecs.size(); ++i) {
    if (i % group_size == 0) {
      for (auto const& e : vecs[i])
        if (e != val + 1) return 1;
    } else {
      for (auto const& e : vecs[i])
        if (e != val) return 1;
    }
    ++val;
  };

  return 0;
}

std::uint32_t test_threaded_detached_stream_fns(std::size_t const& nthreads = 2) {
  using T = double;
  MKN_KUL_DBG_FUNC_ENTER;

  std::vector<ManagedVector<T>> vecs(C, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  ThreadedStreamLauncher launcher{vecs, nthreads};
  launcher
      .host([&](auto i) mutable {
        launcher.streams[i].sync();  // wait for first kernel per stream
        for (auto& e : vecs[i]) e += 1;
      })
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 3; });

  // launched on the launcher's streams before it runs, the first host step syncs on them
  for (std::size_t i = 0; i < datas.size(); ++i) {
    mkn::gpu::DLauncher{NUM}.stream(
        launcher.streams[i], [=, idx = i] __device__() { views[idx][mkn::gpu::idx()] += 1; });
  }

  launcher();

  std::size_t val = 5;  // item i starts at i, + 1 (launched before) + 1 (host) + 3 (dev)
  for (std::size_t i = 0; i < vecs.size(); ++i) {
    for (std::size_t j = 0; j < vecs[i].size(); ++j)
      if (val != vecs[i][j]) return 1;
    ++val;
  };

  return 0;
}

// a host step may launch its own kernel on the item's stream, the device steps after it, queued
//  together as a chain, are ordered behind that kernel; the operations do not commute so any
//  reordering changes the result. The host step prefetches the data it uses, the library does
//  not know what a host step touches
std::uint32_t test_threaded_host_launch(std::size_t const& nthreads = 2) {
  using T = double;
  MKN_KUL_DBG_FUNC_ENTER;

  std::vector<ManagedVector<T>> vecs(C, ManagedVector<T>(NUM, 0));
  for (std::size_t i = 0; i < vecs.size(); ++i) std::fill_n(vecs[i].data(), NUM, i);

  ManagedVector<T*> datas(C);
  for (std::size_t i = 0; i < vecs.size(); ++i) datas[i] = vecs[i].data();
  auto views = datas.data();

  ThreadedStreamLauncher launcher{vecs, nthreads};
  launcher.dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 1; })
      .host([&](auto const i) {
        auto& stream = launcher.streams[i];
        prefetch_host(vecs[i].data(), NUM, stream());
        stream.sync();
        for (auto& e : vecs[i]) e += 1;
        prefetch(vecs[i].data(), NUM, 0, stream());
        auto* view = vecs[i].data();
        mkn::gpu::DLauncher{NUM}.stream(stream, [=] __device__() { view[mkn::gpu::idx()] *= 2; });
      })
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] += 3; })
      .dev([=] __device__(auto const& i) { views[i][mkn::gpu::idx()] *= 10; })();

  for (std::size_t i = 0; i < vecs.size(); ++i)
    for (auto const& e : vecs[i])
      if (e != ((i + 1 + 1) * 2 + 3) * 10) return 1;

  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test()                                                                  //
         + test_threaded(0) + test_threaded()                                    //
         + test_threaded(6)                                                      //
         + test_threaded_group_barrier()                                         //
         + test_threaded_host_group_mutex()                                      //
         + test_threaded_host_group_idx() + test_threaded_detached_stream_fns()  //
         + test_threaded_host_launch(0) + test_threaded_host_launch();
}
