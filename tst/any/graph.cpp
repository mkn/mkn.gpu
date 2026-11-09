// launch overhead: K tiny kernels queued one by one on a stream, or captured once in a Graph and
//  replayed with a single call. The kernels take about a microsecond, so the time per iteration
//  is mostly the CPU side cost of launching them
//  nsys profile --trace=cuda,nvtx <test binary>

#include "mkn/gpu.hpp"
#include "mkn/gpu/trace.hpp"

#include <chrono>

using namespace mkn::gpu;

template <typename T>
using ManagedVector = std::vector<T, ManagedAllocator<T>>;

static constexpr std::size_t NUM = 1024;  // small, kernels in microseconds
static constexpr std::size_t K = 32;      // kernels per iteration
static constexpr std::size_t ITERS = 1000;

void kernels(Stream& stream, float* const p) {
  for (std::size_t k = 0; k < K; ++k)
    DLauncher{NUM}.stream(stream, [=] __device__() { p[idx()] += 1; });
}

std::uint32_t check(ManagedVector<float> const& vec, float const expected) {
  for (auto const& e : vec)
    if (e != expected) {
      KOUT(NON) << "expected " << expected << " got " << e;
      return 1;
    }
  return 0;
}

// capture records without running, each replay runs every kernel once
std::uint32_t test_capture() {
  Stream stream;
  ManagedVector<float> vec(NUM, 0);
  auto* const p = vec.data();

  kernels(stream, p);  // first launches outside the capture, kernel attributes are queried once
  stream.sync();
  if (check(vec, K)) return 1;

  Graph graph;
  graph.capture(stream, [&] { kernels(stream, p); });
  stream.sync();
  if (check(vec, K)) return 1;  // nothing ran
  if (graph.nodes() != K) {
    KOUT(NON) << "graph nodes " << graph.nodes() << " != " << K;
    return 1;
  }

  for (std::size_t i = 0; i < 3; ++i) graph.launch(stream);
  stream.sync();
  return check(vec, K * 4);
}

template <typename F>
double micros_per_iter(F&& f) {
  auto const start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < ITERS; ++i) f();
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
             .count() /
         ITERS;
}

std::uint32_t test_overhead() {
  Stream stream;
  ManagedVector<float> vec(NUM, 0);
  auto* const p = vec.data();
  prefetch(p, NUM, 0, stream());  // resident, no page faults in the timings

  Graph graph;
  graph.capture(stream, [&] { kernels(stream, p); });

  kernels(stream, p);  // warm up both
  graph.launch(stream);
  stream.sync();

  double direct = 0, graphed = 0;
  {
    trace::Range range{"direct"};
    direct = micros_per_iter([&] {
      kernels(stream, p);
      stream.sync();
    });
  }
  {
    trace::Range range{"graph"};
    graphed = micros_per_iter([&] {
      graph.launch(stream);
      stream.sync();
    });
  }
  KOUT(NON) << K << " kernels per iteration, direct " << direct << " us, graph " << graphed
            << " us";
  return check(vec, K * (2 * ITERS + 2));
}

int main() {
  KOUT(NON) << __FILE__;
  return test_capture() + test_overhead();
}
