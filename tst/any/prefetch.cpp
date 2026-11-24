// managed memory used on the device, then the host, then the device again, per stream
//  once migrated by page faults on access, once by prefetching ahead of each use
//  nsys profile --trace=cuda,nvtx --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true
//  <test binary>

#include "mkn/gpu.hpp"
#include "mkn/gpu/trace.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <chrono>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ratio>
#include <tuple>
#include <utility>
#include <vector>
#include <ranges>

MKN_KUL_TEST_DECLARE();

using namespace mkn::gpu;

template <typename T>
using ManagedVector = std::vector<T, ManagedAllocator<T>>;

static constexpr std::size_t NUM = 8 * 1024 * 1024;  // 64MB of doubles per stream
static constexpr std::size_t C = 4;

struct Data {
  Data() : vecs(C, ManagedVector<double>(NUM)), streams(C) {}

  void reset() {  // on the host, so every run starts with all pages there
    for (auto& vec : vecs) std::fill(vec.begin(), vec.end(), 0);
  }

  std::vector<ManagedVector<double>> vecs;
  std::vector<Stream> streams;
};

// device += 1, host += 1, device += 3, every element ends at 5
double run(Data& data, bool const prefetched) {
  data.reset();
  trace::Range range{prefetched ? "prefetched" : "faulted"};
  auto const start = std::chrono::steady_clock::now();

  for (std::size_t i = 0; i < C; ++i) {
    auto* p = data.vecs[i].data();
    if (prefetched) prefetch(p, NUM, 0, data.streams[i]());
    DLauncher{NUM}.stream(data.streams[i], [=] __device__() { p[idx()] += 1; });
    if (prefetched) prefetch_host(p, NUM, data.streams[i]());
  }

  for (std::size_t i = 0; i < C; ++i) {
    data.streams[i].sync();
    for (auto& e : data.vecs[i]) e += 1;
  }

  for (std::size_t i = 0; i < C; ++i) {
    auto* p = data.vecs[i].data();
    if (prefetched) prefetch(p, NUM, 0, data.streams[i]());
    DLauncher{NUM}.stream(data.streams[i], [=] __device__() { p[idx()] += 3; });
  }

  for (auto& stream : data.streams) stream.sync();

  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

void check(Data const& data) {
  MKN_KUL_TEST_CHECK(std::ranges::all_of(data.vecs,
                                         [](auto const& vec) {
                                           return std::ranges::all_of(
                                               vec, [](auto const& e) { return e == 5; });
                                         }),
                     "every value == 5");
}

int main() {
  KOUT(NON) << __FILE__;
  Data data;
  run(data, true);  // warm up, context and allocations

  auto const faulted = run(data, false);
  check(data);  // the values after the run without prefetching
  auto const prefetched = run(data, true);
  check(data);  // the values after the prefetched run

  KOUT(NON) << "faulted " << faulted << " ms, prefetched " << prefetched << " ms";
}
