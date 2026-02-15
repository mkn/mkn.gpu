
template <bool sync = true, typename T>
void __device__ fill_warp_size(T* const t, std::size_t const size, T const val) {
  std::size_t chunk = 0;
  auto const each = size / warpSize;
  for (; chunk < each; ++chunk) t[chunk * warpSize + threadIdx.x] = val;
  if (threadIdx.x < size - (warpSize * each)) t[chunk * warpSize + threadIdx.x] = val;
  if constexpr (sync) __syncthreads();
}
