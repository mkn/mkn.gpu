# Graphs

`mkn::gpu::Graph` records the work queued on a stream once and replays all of it with
a single call. Each launch normally costs microseconds of CPU time. When the kernels
are small, that cost dominates, and replaying a graph removes it.

Source: `Graph` in `inc/mkn/gpu/{cuda,rocm,cpu}/cls.hpp`; test and benchmark in `tst/any/graph.cpp`.

## Usage

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
| `begin(stream)` / `end(stream)` | start / stop capture; `end` instantiates the executable graph |
| `capture(stream, fn)` | `begin`, `fn()`, `end` |
| `launch(stream)` | replay, asynchronous on `stream` |
| `nodes()` | captured node count, one per kernel |
| `empty()` / `reset()` | whether anything has been captured / release it |

## Backends

- CUDA: `cudaStreamBeginCapture` with `cudaStreamCaptureModeThreadLocal`, then
  `cudaStreamEndCapture`, `cudaGraphInstantiateWithFlags` and `cudaGraphLaunch`.
  ThreadLocal mode means a capture on one thread does not affect other threads launching
  on their own streams, such as `ThreadedStreamLauncher` workers.
- ROCm: the same calls with the `hip` prefix.
- CPU: during capture, kernel launches are recorded instead of run (`detail::capture`, thread
  local), and `launch` runs them in order. Only kernel launches are recorded. Copies, and a
  `Graph::launch` made during a capture, run immediately.

## Caveats

- Kernel arguments are fixed at capture time. Pointers are fine as long as the memory
  they point to stays allocated, but a host value that changes between replays (such as a
  time step `dt`) needs a new capture. This is why inference engines capture one graph
  per batch size bucket and pad requests to fit.
- Launch each kernel once before capturing it. The first launch queries the kernel's
  attributes (`device_kernel_bx_threads`, `fit_shared`), and doing that beforehand keeps
  the capture to stream work only. `tst/any/graph.cpp` does this.
- Synchronous calls (`sync`, `cudaMemcpy`, allocation) are not allowed during a capture.
  They invalidate it, and the call or `end` reports the error.
- `StreamLauncher` / `ThreadedStreamLauncher` don't use graphs yet. They run their
  pipeline only once, so a graph would never be replayed. The launcher has to become
  re-runnable first.

## Results: `tst/any/graph.cpp`

32 kernels per iteration, each doing `+= 1` on 1024 floats in managed memory prefetched
to the device. Each iteration launches and syncs; 1000 iterations.

```bash
mkn build -x res/mkn/clang_cuda test -p cuda -M tst/any/graph.cpp run
nsys profile --trace=cuda,nvtx -o graph bin/cuda/test/*_graph.cpp
nsys stats --report nvtx_sum --report cuda_api_sum graph.nsys-rep
```

| | direct | graph |
|---|---|---|
| RTX 5080, CUDA 13.5 | 57–62 us | 28–30 us |
| AMD Radeon iGPU (9800X3D), ROCm 7.1.1 | 69 us | 70 us |

nsys `cuda_api_sum` on the 5080:

```
 Num Calls  Avg (ns)  Name
    32,128   1,933.6  cudaLaunchKernel
     1,004   1,720.7  cudaGraphLaunch
```

- A direct iteration spends about 32 × 1.9 us ≈ 62 us of CPU time launching, more than
  the GPU needs to run the kernels.
- A graph iteration takes one 1.7 us call. What remains, about 0.9 us per kernel, is GPU
  time running the kernels back to back.
- The NVTX ranges `mkn.gpu:direct` and `mkn.gpu:graph` separate the two phases. Under
  nsys, the direct phase slows down further (85 ms vs 27 ms), because tracing adds a cost
  to every API call.
- On the ROCm iGPU there is no gain: HIP replays the graph as ordinary stream launches
  there, so it pays the same per-launch cost. Measure on each platform before relying on
  graphs.
