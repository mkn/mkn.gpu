// coalesced vs strided global memory access, an N x N float matrix transposed
//  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis_Tables <test binary>
//
//  copy_kernel      reads and writes coalesced, the bandwidth ceiling
//  naive            reads coalesced, writes strided by N, each warp store touches 32 sectors
//  tiled            a 32 x 32 tile staged in shared memory, reads and writes coalesced
//  tiled_conflicts  as tiled without the padding column, the column reads from shared memory
//                   all hit the same bank, 32 way bank conflicts

#define MKN_GPU_FN_PER_NS 1
#include "mkn/gpu/cuda.hpp"

using namespace mkn::gpu::cuda;

static constexpr std::uint32_t N = 8192;             // 256MB per matrix
static constexpr std::uint32_t TILE = 32, ROWS = 8;  // blocks of 32 x 8, each thread 4 rows

__global__ void copy_kernel(float* out, float const* in) {
  auto const x = blockIdx.x * TILE + threadIdx.x;
  auto const y = blockIdx.y * TILE + threadIdx.y;
  for (std::uint32_t j = 0; j < TILE; j += ROWS) out[(y + j) * N + x] = in[(y + j) * N + x];
}

__global__ void naive(float* out, float const* in) {
  auto const x = blockIdx.x * TILE + threadIdx.x;
  auto const y = blockIdx.y * TILE + threadIdx.y;
  for (std::uint32_t j = 0; j < TILE; j += ROWS) out[x * N + (y + j)] = in[(y + j) * N + x];
}

template <std::uint32_t PAD>
__device__ void tiled_t(float* out, float const* in) {
  __shared__ float tile[TILE][TILE + PAD];
  auto x = blockIdx.x * TILE + threadIdx.x;
  auto y = blockIdx.y * TILE + threadIdx.y;
  for (std::uint32_t j = 0; j < TILE; j += ROWS)
    tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;  // swapped block, the output tile
  y = blockIdx.x * TILE + threadIdx.y;
  for (std::uint32_t j = 0; j < TILE; j += ROWS)
    out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void tiled(float* out, float const* in) { tiled_t<1>(out, in); }
__global__ void tiled_conflicts(float* out, float const* in) { tiled_t<0>(out, in); }

template <typename K>
std::uint32_t run(K kernel, bool const transposed, std::vector<float> const& host,
                  DeviceMem<float>& in, DeviceMem<float>& out) {
  out.fill_n(-1, out.size());  // host has no negatives, a write the kernel skips is caught
  GLauncher{dim3{N / TILE, N / TILE}, dim3{TILE, ROWS}}(kernel, out, in).sync();
  auto const res = out();
  for (std::size_t y = 0; y < N; y += 61)  // sampled, a full check is slow on the host
    for (std::size_t x = 0; x < N; x += 59)
      if (res[y * N + x] != (transposed ? host[x * N + y] : host[y * N + x])) return 1;
  return 0;
}

int main() {
  KOUT(NON) << __FILE__;
  std::vector<float> host(std::size_t{N} * N);
  for (std::size_t i = 0; i < host.size(); ++i) host[i] = i;
  DeviceMem<float> in{host}, out(host.size());

  return run(copy_kernel, false, host, in, out)  //
         + run(naive, true, host, in, out)       //
         + run(tiled, true, host, in, out)       //
         + run(tiled_conflicts, true, host, in, out);
}
