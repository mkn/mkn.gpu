# mkn.gpu

CUDA/HIP C++20 convenience wrappers

[README](https://raw.githubusercontent.com/mkn/mkn.gpu/master/README.noformat)

## Example

The same lambda run over host data and over managed data, see [tst/any/readme.cpp](tst/any/readme.cpp).

```cpp
#include "mkn/gpu.hpp"

using namespace mkn::gpu;

static constexpr std::size_t N = 1'000'000;

// host data: plain host memory, the device cannot read it, so copy in and out
std::vector<float> host(N, 1);
DeviceMem<float> dev{host};  // allocate on the device and copy host -> device
DLauncher{N}([] __device__(float* d) { d[idx()] += 1; }, dev).sync();  // dev is passed as float*
host = dev();  // copy device -> host

// managed data: the same pointer is valid on host and device, pages migrate on access
std::vector<float, ManagedAllocator<float>> managed(N, 1);
DLauncher{managed}([] __device__(float* d) { d[idx()] += 1; }, managed).sync();  // no copies
```

What happens:

| | host data (`std::vector` + `DeviceMem`) | managed data (`ManagedAllocator`) |
|---|---|---|
| memory | separate host and device allocations | one allocation, visible to both |
| copies | explicit: `DeviceMem{host}` in, `dev()` out | none written, the driver migrates pages |
| argument to the lambda | `DeviceMem` becomes its device `float*` | the vector becomes its `float*` |
| host access after `sync()` | only after copying back | direct, `managed[i]` |
| cost | two full copies over PCIe | page faults on first touch each side, see `doc/tracing` |

A plain `std::vector` cannot be passed to a launch directly, its memory is on the host only.

`DLauncher{N}` guesses the launch from the size: blocks of the device maximum (or
`MKN_GPU_BX_THREADS`), reduced to the kernel's own limit when its register or shared memory use
is high, and enough blocks to cover `N`. Each element runs once, `idx()` is its index.

On the CPU backend the same code runs on the host: `DeviceMem` and managed memory are host memory
and the lambda runs once per element in a loop.

## Backend

Resolved at compile time from the headers available, see [inc/mkn/gpu/def.hpp](inc/mkn/gpu/def.hpp):

| headers found | backend |
|---|---|
| `cuda_runtime.h` only | CUDA |
| `hip/hip_runtime.h` only | ROCm |
| neither | CPU |
| both | compile error, define one |

Defines override the detection:

| define | use |
|---|---|
| `-DMKN_GPU_CUDA=1` | CUDA, e.g. when both headers are found |
| `-DMKN_GPU_ROCM=1` | ROCm, e.g. when both headers are found |
| `-DMKN_GPU_CPU=1` | CPU while GPU headers are available, e.g. to test without a device |
