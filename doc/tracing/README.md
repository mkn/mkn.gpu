# Tracing output examples

Example output of `tst/any/trace.cpp` under each vendor profiler.
Annotations come from `inc/mkn/gpu/trace.hpp` (`trace::Range`, `trace::mark`).

## CUDA / Nsight Systems — [nsys.txt](nsys.txt)

```bash
mkn build -x res/mkn/clang_cuda test -p cuda_trace
nsys profile --trace=cuda,nvtx -o trace bin/cuda_trace/test/*_trace.cpp
nsys stats --report nvtx_sum --report nvtx_gpu_proj_sum --report cuda_gpu_kern_sum trace.nsys-rep
```

- Ranges are in the `mkn.gpu` NVTX domain, shown as `mkn.gpu:<name>`.
- `nvtx_sum` is host (CPU) time inside each range.
- `nvtx_gpu_proj_sum` projects each range onto the GPU work launched within it.
  The first `launch` projects to ~75us vs ~3us for the others: the kernel touches
  managed memory, so the first launch pays the page-fault migration.

## ROCm / rocprofv3 — [rocprofv3_marker_api_trace.csv](rocprofv3_marker_api_trace.csv), [rocprofv3_kernel_trace.csv](rocprofv3_kernel_trace.csv)

```bash
mkn build -x res/mkn/hipcc test -p rocm_trace
rocprofv3 --marker-trace --kernel-trace --output-format csv -d out -- bin/rocm_trace/test/*_trace.cpp
```

- roctx requires `-DMKN_GPU_TRACE=1` and linking `rocprofiler-sdk-roctx`, as the `rocm_trace` profile does.
- `trace::mark` shows as a zero-length entry (`before launch`).
- Correlate markers with kernels by timestamp.

## CUDA / HIP Graphs — [graphs.md](graphs.md)

Capturing launches once and replaying them: per-launch CPU cost, measured with nsys.
