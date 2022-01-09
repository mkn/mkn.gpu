
template <typename Float, bool GPU = false>
struct DevClass : mkn::gpu::DeviceClass<GPU> {
  using Super = mkn::gpu::DeviceClass<GPU>;
  using gpu_t = DevClass<Float, true>;

  template <typename T>
  using container_t = typename Super::template container_t<T>;

  DevClass(DevClass const&) = delete;
  auto& operator=(DevClass const&) = delete;

  DevClass(std::uint32_t nbr) : data{nbr} {}

  DevClass(std::vector<Float> const& in) : data{in} {}

  auto operator()() __host__ { return Super::template alloc<gpu_t>(data); }

  auto& operator[](int i) __device__ { return data[i]; }
  auto const& operator[](int i) const __device__ { return data[i]; }

  container_t<Float> data;
};
