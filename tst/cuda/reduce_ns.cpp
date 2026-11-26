#define MKN_GPU_FN_PER_NS 1
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

// MKN_GPU_FN_PER_NS: the backend header is used directly, not mkn/gpu.hpp, and reduce is
//  mkn::gpu::cuda::reduce
// IWYU pragma: no_include "mkn/gpu.hpp"
#include "mkn/gpu/cuda.hpp"  // IWYU pragma: keep
#include "mkn/gpu/reduce.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

MKN_KUL_TEST_DECLARE();

namespace gpu = mkn::gpu::cuda;
namespace op = mkn::gpu::reduce_op;

int main() {
  KOUT(NON) << __FILE__;
  std::size_t constexpr n = 100003;
  std::vector<std::int64_t> host(n);
  std::iota(host.begin(), host.end(), std::int64_t{-50000});
  gpu::DeviceMem<std::int64_t> dev(host);
  auto const sum = std::accumulate(host.begin(), host.end(), std::int64_t{7});
  MKN_KUL_TEST_CHECK(gpu::reduce(dev, std::int64_t{7}, op::Sum{}) == sum, "sum");
  MKN_KUL_TEST_CHECK(gpu::reduce(dev.data(), n, std::int64_t{0}, op::Min{}) == -50000, "min");
  MKN_KUL_TEST_CHECK(gpu::reduce(dev, std::int64_t{0}, op::Max{}) == 50002, "max");
  MKN_KUL_TEST_CHECK(gpu::reduce(dev.data(), 0, std::int64_t{3}, op::Sum{}) == 3, "empty");
}
