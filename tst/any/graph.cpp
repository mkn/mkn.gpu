// launch overhead: K tiny kernels queued one by one on a stream, or captured once in a Graph and
//  replayed with a single call. The kernels take about a microsecond, so the time per iteration
//  is mostly the CPU side cost of launching them. Then the same for a pipeline of steps per data
//  item, GraphLauncher against the stream launchers
//  nsys profile --trace=cuda,nvtx <test binary>

#include "mkn/gpu.hpp"
#include "mkn/gpu/trace.hpp"
#include "mkn/gpu/graph.hpp"
#include "mkn/gpu/stream.hpp"

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
  if (!CompileFlags::withCPU && graph.nodes() != K) {
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

template <typename F>
double millis_total(F&& f) {
  auto const start = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
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

static constexpr std::size_t C = 4;       // data items, one stream each
static constexpr std::size_t STEPS = 16;  // device steps per item
static constexpr std::size_t RUNS = 200;

struct Data {
  Data() : vecs(C, ManagedVector<float>(NUM, 0)), datas(C) {
    for (std::size_t i = 0; i < C; ++i) datas[i] = vecs[i].data();
  }
  std::vector<ManagedVector<float>> vecs;
  ManagedVector<float*> datas;
};

template <typename Launcher>
void add_steps(Launcher& launcher, Data& data) {
  auto views = data.datas.data();
  for (std::size_t s = 0; s < STEPS; ++s)
    launcher.dev([=] __device__(auto const i) { views[i][idx()] += 1; });
}

std::uint32_t check(Data const& data, float const expected) {
  std::uint32_t errors = 0;
  for (auto const& vec : data.vecs) errors += check(vec, expected);
  return errors;
}

// device steps around host steps, replayed: host steps see the device steps before them
std::uint32_t test_graph_launcher() {
  Data data;
  auto views = data.datas.data();
  std::vector<float> seen(C, 0);
  {
    GraphLauncher launcher{data.vecs};
    launcher.dev([=] __device__(auto const i) { views[i][idx()] += 1; })
        .host([&](auto const i) { seen[i] = data.vecs[i][NUM - 1]; })
        .dev([=] __device__(auto const i) { views[i][idx()] += 2; });
    for (std::size_t r = 0; r < 3; ++r) launcher();
    launcher.sync();
    if (!CompileFlags::withCPU && launcher.graph.nodes() < C * 3) {
      KOUT(NON) << "graph nodes " << launcher.graph.nodes();
      return 1;
    }
  }
  for (auto const& s : seen)
    if (s != 7) {  // third run: 3 + 3 + 1
      KOUT(NON) << "host step saw " << s;
      return 1;
    }
  return check(data, 9);
}

std::uint32_t test_launchers() {
  double stream = 0, threaded = 0, graphed = 0;
  {
    Data data;
    trace::Range range{"StreamLauncher"};
    stream = millis_total([&] {
      for (std::size_t r = 0; r < RUNS; ++r) {
        StreamLauncher launcher{data.vecs};
        add_steps(launcher, data);
        launcher();
      }
    });
    if (check(data, STEPS * RUNS)) return 1;
  }
  {
    Data data;
    trace::Range range{"ThreadedStreamLauncher"};
    threaded = millis_total([&] {
      for (std::size_t r = 0; r < RUNS; ++r) {
        ThreadedStreamLauncher launcher{data.vecs, 0};
        add_steps(launcher, data);
        launcher();
      }
    });
    if (check(data, STEPS * RUNS)) return 1;
  }
  {
    Data data;
    GraphLauncher launcher{data.vecs};
    add_steps(launcher, data);
    launcher();  // capture outside the timing, it is once
    launcher.sync();
    trace::Range range{"GraphLauncher"};
    graphed = millis_total([&] {
      for (std::size_t r = 1; r < RUNS; ++r) {
        launcher();
        launcher.sync();
      }
    });
    if (check(data, STEPS * RUNS)) return 1;
  }
  KOUT(NON) << C << " items x " << STEPS << " device steps, ms per run: StreamLauncher "
            << stream / RUNS << ", ThreadedStreamLauncher(0) " << threaded / RUNS
            << ", GraphLauncher " << graphed / (RUNS - 1);
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  return test_capture() + test_overhead() + test_graph_launcher() + test_launchers();
}
