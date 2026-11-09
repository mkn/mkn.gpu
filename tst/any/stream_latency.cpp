// scheduling overhead of the stream launchers, many small device steps per data item
//  the kernels take microseconds, so the run time is mostly the time between steps

#include "mkn/gpu.hpp"
#include "mkn/gpu/stream.hpp"
#include "mkn/kul/log.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ratio>
#include <vector>

using namespace mkn::gpu;

template <typename T>
using ManagedVector = std::vector<T, ManagedAllocator<T>>;

static constexpr std::size_t NUM = 1024;  // small, kernels in microseconds
static constexpr std::size_t C = 4;       // data items, one stream each
static constexpr std::size_t STEPS = 16;  // device steps per item, in pairs around a host step

struct Data {
  Data() : vecs(C, ManagedVector<float>(NUM, 0)), datas(C) {
    for (std::size_t i = 0; i < C; ++i) datas[i] = vecs[i].data();
  }
  std::vector<ManagedVector<float>> vecs;
  ManagedVector<float*> datas;
};

// with_host: a host step after every 2 device steps, which migrates the managed pages back and
//  forth, without: device steps only, the time is the scheduling between them
template <bool with_host, typename Launcher>
void add_steps(Launcher& launcher, Data& data) {
  auto views = data.datas.data();
  for (std::size_t s = 0; s < STEPS / 2; ++s) {
    launcher.dev([=] __device__(auto const i) { views[i][idx()] += 1; });
    launcher.dev([=] __device__(auto const i) { views[i][idx()] += 1; });
    if constexpr (with_host) launcher.host([&](auto const i) { data.vecs[i][0] += 0; });
  }
}

std::uint32_t check(Data const& data) {
  for (auto const& vec : data.vecs)
    for (auto const& e : vec)
      if (e != STEPS) return 1;
  return 0;
}

template <typename F>
double millis(F&& f) {
  auto const start = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

template <bool with_host>
std::uint32_t test_stream_launcher() {
  Data data;
  auto const ms = millis([&] {
    StreamLauncher launcher{data.vecs};
    add_steps<with_host>(launcher, data);
    launcher();
  });
  KOUT(NON) << "StreamLauncher host=" << with_host << " " << ms << " ms";
  return check(data);
}

template <bool with_host>
std::uint32_t test_threaded_stream_launcher(std::size_t const nthreads) {
  Data data;
  auto const ms = millis([&] {
    ThreadedStreamLauncher launcher{data.vecs, nthreads};
    add_steps<with_host>(launcher, data);
    launcher();
  });
  KOUT(NON) << "ThreadedStreamLauncher(" << nthreads << ") host=" << with_host << " " << ms
            << " ms";
  return check(data);
}

int main() {
  KOUT(NON) << __FILE__;
  test_threaded_stream_launcher<false>(0);  // warm up
  return test_stream_launcher<true>() + test_stream_launcher<false>() +
         test_threaded_stream_launcher<true>(0) + test_threaded_stream_launcher<true>(2) +
         test_threaded_stream_launcher<false>(0) + test_threaded_stream_launcher<false>(2);
}
