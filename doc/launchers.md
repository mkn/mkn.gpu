# Launchers

Single launches:

| | launches | grid/block | use for |
|---|---|---|---|
| `DLauncher` | a `__device__` lambda or function | guessed from a size, or given | most kernels |
| `GLauncher` | a `__global__` function | guessed from a size, or given | existing CUDA/HIP kernels |

Pipelines, a list of steps run for each data item:

| | runs | items in parallel | host steps in parallel | can be run again |
|---|---|---|---|---|
| `StreamLauncher` | one thread, polling | yes | no | no |
| `ThreadedStreamLauncher` | worker threads | yes | yes, one per thread | no |
| `GraphLauncher` | one graph launch | yes | no | yes |

All launches are asynchronous. Call `sync()` to wait for them.

Host steps and managed memory: in every pipeline launcher, a host step of one item can run
while kernels of other items are still running. Touching managed memory from the host at
that point is only allowed on devices with concurrent managed access
(`getConcurrentManagedAccess()`, true on Linux with a Pascal or newer GPU). Without it, for
example on Windows, any host access to managed memory while a kernel runs, on any stream,
crashes. On such devices, host steps must not touch managed memory: copy what they need
to or from host or pinned memory in a device step. `prefetch()` does nothing on these
devices.

## DLauncher

Source: `inc/mkn/gpu/any/inc/launchers.ipp`

```cpp
DLauncher{N}([] __device__(float* d) { d[idx()] += 1; }, managed).sync();
DLauncher{vec}.stream(stream, [=] __device__() { p[idx()] += 1; });  // size from vec.size()
DLauncher{dim3{blocks}, dim3{threads}}(fn, args...);                 // fixed
```

- `DLauncher{size}` or `DLauncher{container}` (Guess) picks the block size and enough
  blocks to cover `size`.
  - The block size is the device maximum, reduced to the generated kernel's own limit when
    its register or shared memory use is high.
  - Threads past `size` don't run the function, so `idx()` is always in range.
- `DLauncher{grid, block}` and the other dimension constructors (Fixed) launch exactly what
  you give, with no bounds check.
- Arguments are converted to what the device can use: a `DeviceMem` becomes its device
  pointer, a managed vector its `data()`. A plain `std::vector` is rejected.
- `ds` sets the dynamic shared memory size. A kernel needing more than the default limit is
  opted in automatically, per kernel and device.
- `.stream(s, ...)` launches on a `Stream` instead of the default stream.
- Guessed launches must fit in 32-bit indexing, and launching one that doesn't throws.

## GLauncher

```cpp
GLauncher{WIDTH, HEIGHT, TPB_X, TPB_Y}(vector_inc0<float>, devA);  // fixed, 2D
GLauncher{NUM}(kernel, v.data(), NUM).sync();                      // guessed

static auto const bx = kernel_bx_threads<heavy>();  // per kernel, once
GLauncher{NUM}.bx(bx)(heavy, out, in, NUM, 4u);     // guessed, within the kernel's limit
```

- It has the same constructors, `ds`, `.stream()` and `sync()` as `DLauncher`, but takes
  your own `__global__` function.
- Guessed launches are not bounds checked: the kernel must check `idx() < size` itself.
- The block size of a guessed launch is not reduced to the kernel's limit unless you pass
  it with `.bx(kernel_bx_threads<kernel>())`. A kernel with high register use otherwise
  fails with "too many resources requested for launch". See `tst/any/occupancy.cpp`.

Use `DLauncher` unless you already have the `__global__` function, or need
`__launch_bounds__` or a specific kernel signature.

## StreamLauncher

Source: `inc/mkn/gpu/stream.hpp`

```cpp
StreamLauncher launcher{datas};  // one Stream per item
launcher.dev([=] __device__(auto const i) { views[i][idx()] += 1; })  // i: item, idx(): element
    .host([&](auto const i) { /* on the host, after item i's device steps */ });
launcher();  // returns when every item has finished every step
```

- Each item runs every step in order, and different items run independently.
- A `dev` step is one `DLauncher{datas[i]}` kernel on item `i`'s stream, so items must
  have a `size()`.
- A `host` step runs on the calling thread.
- One thread advances all items, polling for completion every 1 ms. Each device step
  therefore takes at least about 1 ms, and host steps of different items run one after
  another.
- The pipeline runs once. Build a new launcher for the next run.

Simple and easy to reason about, but slow. Use `ThreadedStreamLauncher` instead unless
you need everything to happen on one thread.

## ThreadedStreamLauncher

```cpp
ThreadedStreamLauncher launcher{datas, /*n_threads=*/2};
launcher.dev(...)
    .host(...)                                  // on a worker thread
    .barrier()                                  // every item reaches this step first
    .group_barrier(4)                           // every item in its group of 4
    .host_group_mutex(4, [&](auto const i) {})  // one item of each group at a time
    .host_group_idx(4, 0, [&](auto const i) {}) // only item 0 of each group
    .dev(...);
launcher();  // the calling thread works too, then joins the workers
launcher.print_times(1e6);  // per item, per step, in ms
```

- `n_threads` is the number of extra worker threads. The calling thread works too, inside
  `launcher()` / `join()`, so `n_threads + 1` threads take whichever item is ready next,
  and up to that many host steps of different items run in parallel. `n_threads = 0` is
  the calling thread alone.
- A step finishing wakes a thread: device steps signal through a completion callback on
  the item's stream. There is no polling.
- Consecutive device steps are queued together with one callback at the end of the run,
  rather than one callback and handoff per step.
- Barrier and group steps let items wait for each other or take turns.
- Each step of each item gets an NVTX/roctx range, and is timed (`print_times`,
  `dump_times`).
- The pipeline runs once. Build a new launcher for the next run.

Use this for pipelines with real host work, steps that need items to wait for each other,
or when you want a per-step timing breakdown.

## GraphLauncher

Source: `inc/mkn/gpu/graph.hpp`, details and numbers in [tracing/graphs.md](tracing/graphs.md)

```cpp
GraphLauncher launcher{datas};
launcher.dev(...).host(...).dev(...);
for (std::size_t t = 0; t < steps; ++t) launcher();  // first call records, the rest replay
launcher.sync();
```

- The first call records every step of every item into one CUDA/HIP graph, and every call
  replays it with a single launch.
- Each item is a parallel branch, and its steps run in order.
- Host steps become host nodes in the graph. They run in stream order but all on one
  internal thread, and must not call the GPU API.
- It can be called any number of times. Adding a step, or calling `reset()`, makes the
  next call record again.
- On CPU every call runs the steps.

## Graph or stream launchers

| | GraphLauncher | ThreadedStreamLauncher |
|---|---|---|
| 4 items x 16 small device steps, RTX 5080 | 0.02 ms | 0.15 ms |
| same, ROCm iGPU | 0.08 ms | 2.7 ms |
| 4 items x one 10 ms host step | 40 ms | 10 ms (`n_threads = 3`, 4 threads working) |
| CPU cost per run | one graph launch, ~5 us | a launch per kernel, a callback per item, ~200 us |
| kernel arguments, sizes, pointers | fixed when recorded | read on every run |
| barriers, group steps | no | yes |
| per-step trace ranges and timings | no, one range per run | yes |
| first run | slower, recording ~250 us | normal |

Use `GraphLauncher` when:

- the same pipeline runs many times, such as every time step;
- the steps are mostly small kernels, so launching them costs more than running them;
- nothing a device step reads from the host changes between runs. That includes item
  sizes and allocations: a vector that resizes or reallocates leaves the graph with a
  stale size or a dangling pointer, with no error.

Use `ThreadedStreamLauncher` when:

- host steps do real work, since they run in parallel on its threads but one after
  another in a graph;
- steps change values, sizes or allocations between runs;
- you need barriers or group steps;
- you are profiling and want each step timed separately;
- the pipeline runs once, where recording a graph would cost more than it saves.

If unsure, start with `ThreadedStreamLauncher`. Switch to `GraphLauncher` when a profile
shows CPU launch overhead dominating (nsys `cuda_api_sum`: `cudaLaunchKernel` time close
to the kernels' own time), and only once the pipeline's inputs are fixed between runs.

On CPU the choice makes no difference to correctness. On a GPU, a `GraphLauncher` reading
a host value that changes between runs is wrong, while on CPU the same code still passes
its tests.
