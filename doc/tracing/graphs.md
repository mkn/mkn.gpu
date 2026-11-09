# Graphs

A graph records GPU work once and replays all of it with a single call. Each launch
normally costs microseconds of CPU time. When the kernels are small, that cost dominates,
and replaying a graph removes it.

- `GraphLauncher` (`inc/mkn/gpu/graph.hpp`): a pipeline of steps per data item, like
  `StreamLauncher`, recorded on the first call and replayed by every call after.
- `Graph` (`inc/mkn/gpu/{cuda,rocm,cpu}/cls.hpp`): the building block; records whatever a
  function queues on one or more streams.

Tests and benchmarks: `tst/any/graph.cpp`.

## GraphLauncher

```cpp
GraphLauncher launcher{datas};  // one stream per item
launcher.dev([=] __device__(auto const i) { views[i][idx()] += 1; })
    .host([&](auto const i) { seen[i] = data.vecs[i][0]; })
    .dev([=] __device__(auto const i) { views[i][idx()] += 2; });

for (std::size_t t = 0; t < steps; ++t) launcher();  // first call records, the rest replay
launcher.sync();
```

- The whole pipeline, every item and every step, is one graph and one `cudaGraphLaunch`
  per call.
- Each item is a parallel branch on its own stream, and its steps run in order.
- Host steps become host nodes (`cudaLaunchHostFunc` during capture). They run in stream
  order after the item's earlier steps, with no handoff to a worker thread. They must not
  call the GPU API.
- Unlike `StreamLauncher` / `ThreadedStreamLauncher`, which run their pipeline once,
  `GraphLauncher` can be called any number of times.
- Calls are asynchronous; `sync()` waits for completion. Adding a step, or calling
  `reset()`, makes the next call record again.
- On CPU every call runs the steps.
- Barrier and group steps are not supported.
- Device steps of different items run at the same time: 4 items each running a ~10 ms
  kernel took 10.5 ms. Host nodes do not: CUDA runs them all on one internal thread, so
  4 items each with a 10 ms host step took 40 ms. `ThreadedStreamLauncher` ran the same
  host steps in 20 ms with `n_threads = 1` and 10 ms with `n_threads = 3`. `n_threads`
  counts extra threads and the calling thread works too, so that is 2 and 4 threads.
  When host steps do real work, `ThreadedStreamLauncher` is the better choice.

## Graph

```cpp
Stream stream;
Graph graph;
graph.capture(stream, [&] {  // recorded, nothing runs yet
  for (std::size_t k = 0; k < K; ++k)
    DLauncher{n}.stream(stream, [=] __device__() { p[idx()] += 1; });
});

for (std::size_t i = 0; i < steps; ++i) graph.launch(stream);  // asynchronous
stream.sync();
```

| Member | |
|---|---|
| `capture(stream, fn)` | record what `fn` queues on `stream` |
| `capture(streams, fn)` | record what `fn` queues on any of `streams`, each one a parallel branch |
| `launch(stream)` | replay, asynchronous; for a multi-stream capture, sync the first stream |
| `nodes()` | node count in the graph, 0 on CPU |
| `empty()` / `reset()` | whether anything has been captured / release it |
| `launch_host_fn(stream, fn, data)` | free function, `fn(data)` on the host in stream order |

Backends:

- CUDA: `cudaStreamBeginCapture` with `cudaStreamCaptureModeThreadLocal`, then
  `cudaStreamEndCapture`, `cudaGraphInstantiateWithFlags` and `cudaGraphLaunch`.
  ThreadLocal mode means a capture on one thread does not affect other threads launching
  on their own streams.
- Multi-stream capture: the other streams join by waiting on an event recorded on the
  first stream, and the first stream waits on an event from each of them before the
  capture ends.
- ROCm: the same calls with the `hip` prefix.
- CPU: nothing is recorded. `capture` stores `fn` and `launch` calls it again, so unlike on
  a GPU, host code in `fn` runs on every launch and kernel arguments are read again.

## Caveats

- Kernel arguments are fixed at capture time. Pointers are fine as long as the memory
  they point to stays allocated, but a host value that changes between replays (such as a
  time step `dt`) is not seen by the replay. There are two ways to change it:
  - Record again (`reset()`, a few hundred microseconds).
  - Update the instantiated graph in place, which is much cheaper. One option is
    `cudaGraphExecKernelNodeSetParams` per kernel node. Another is
    `cudaGraphExecUpdate` with a newly captured graph of the same shape, which skips
    instantiation; HIP has both. `Graph` / `GraphLauncher` don't use either yet.

  Inference engines hit the same limit with batch shapes: they capture one graph per
  batch size bucket and pad requests to fit.
- Synchronous calls are not allowed during a capture: synchronising the capturing stream
  (`sync`), synchronous `cudaMemcpy` (it runs on the legacy default stream, which
  implicitly synchronises with the capturing stream), and `cudaMalloc` / `cudaFree`. They
  invalidate the capture, and the call or the end of the capture reports the error. Their
  asynchronous versions on the capturing stream are recorded instead: `cudaMemcpyAsync`
  as a memcpy node, and `cudaMallocAsync` / `cudaFreeAsync` as memory nodes.
- Recording takes a few hundred microseconds, so graphs only pay off when replayed many
  times.

## Results: `tst/any/graph.cpp`

```bash
mkn build -x res/mkn/clang_cuda test -p cuda -M tst/any/graph.cpp
nsys profile --trace=cuda,nvtx -o graph bin/cuda/test/*_graph.cpp
nsys stats --report nvtx_sum --report cuda_api_sum --report cuda_gpu_kern_sum graph.nsys-rep
```

Machine: RTX 5080 with CUDA 13.5; AMD Radeon iGPU (9800X3D) with ROCm 7.1.1.

### One stream: 32 kernels launched directly vs one Graph

Each kernel does `+= 1` on 1024 floats in managed memory prefetched to the device. Each
iteration launches and syncs; 1000 iterations.

| per iteration | direct | Graph |
|---|---|---|
| RTX 5080 | 57–62 us | 28–30 us |
| ROCm iGPU | 68 us | 65–70 us |

### Pipeline: 4 items x 16 device steps, 200 runs

| per run | ThreadedStreamLauncher | GraphLauncher |
|---|---|---|
| RTX 5080 | 0.15 ms | 0.020–0.025 ms |
| ROCm iGPU | 2.7 ms | 0.08 ms |
| CPU | 0.18 ms | 0.17 ms |

`ThreadedStreamLauncher` here has `n_threads = 0`, so only the calling thread works. It is
rebuilt for every run, which is how it is used since it runs only once. The test also
times `StreamLauncher` (17–21 ms), but that number is its 1 ms polling between steps, not
the GPU.

nsys on the 5080:

```
 Num Calls  Avg (ns)  Name
    57,800   2,135.4  cudaLaunchKernel    (median 1,954)
     1,207   2,457.9  cudaGraphLaunch     (median 1,593)
         4     593.5  cudaLaunchHostFunc  (capture only)

 Instances  Med (ns)  Range
       200   204,330  mkn.gpu:ThreadedStreamLauncher::join
       204     5,400  mkn.gpu:GraphLauncher
         2   250,477  mkn.gpu:GraphLauncher::capture
```

- A direct iteration spends about 32 × 2 us ≈ 62 us of CPU time launching (median
  `cudaLaunchKernel` 1.95 us, average 2.1 us), more than the GPU needs to run the
  kernels (~0.6 us each).
- A graph replay takes one ~1.6 us call. The CPU side of a `GraphLauncher` run is 5.4 us,
  against 204 us for `ThreadedStreamLauncher::join`, which launches 64 kernels and handles
  a completion callback per item.
- A `GraphLauncher` run (~20 us) is GPU time: 16 kernels in order on each of 4
  concurrent branches.
- On ROCm, a single-stream graph gains nothing. That is consistent with HIP replaying the
  graph as ordinary stream launches, with the same per-launch cost, but it hasn't been
  checked with rocprofv3. `GraphLauncher` still wins by a wide margin (2.7 ms → 0.08 ms).
  That suggests the cost there is the completion callbacks (`hipStreamAddCallback`) and the
  handoff between threads, which the graph removes; this hasn't been profiled either.
  Measure on each platform before relying on graphs.
