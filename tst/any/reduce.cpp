
#include "mkn/gpu.hpp"
#include "mkn/gpu/reduce.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

MKN_KUL_TEST_DECLARE();

using namespace mkn::gpu;

// sizes around the block (256) and grid (1024 blocks: 262144 values a stride) limits
static std::size_t constexpr SIZES[] = {1, 2, 255, 256, 257, 1000, 262144, 262145, 3000017};

// integers: every grouping gives the host's sum, min and max exactly
void test_integers() {
  for (auto const n : SIZES) {
    std::vector<std::int64_t> h(n);
    for (std::size_t i = 0; i < n; ++i)
      h[i] = static_cast<std::int64_t>((i * 2654435761u) % 1000003u) - 500000;
    DeviceMem<std::int64_t> d(h);
    auto const sum = std::accumulate(h.begin(), h.end(), std::int64_t{7});
    MKN_KUL_TEST_CHECK(reduce(d, std::int64_t{7}, reduce_op::Sum{}) == sum,
                       "sum n=" + std::to_string(n));
    MKN_KUL_TEST_CHECK(reduce(d, std::numeric_limits<std::int64_t>::max(), reduce_op::Min{}) ==
                           *std::min_element(h.begin(), h.end()),
                       "min n=" + std::to_string(n));
    MKN_KUL_TEST_CHECK(reduce(d, std::numeric_limits<std::int64_t>::lowest(), reduce_op::Max{}) ==
                           *std::max_element(h.begin(), h.end()),
                       "max n=" + std::to_string(n));
  }
}

// doubles holding integers below 2^53: sums exact in any order
void test_doubles_exact() {
  for (auto const n : SIZES) {
    std::vector<double> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = static_cast<double>(i % 97) - 48.;
    DeviceMem<double> d(h);
    auto const sum = std::accumulate(h.begin(), h.end(), 0.5);
    MKN_KUL_TEST_CHECK(reduce(d, 0.5, reduce_op::Sum{}) == sum, "sum n=" + std::to_string(n));
  }
}

// floats: min / max exact, the sum within round off of the host's, and the same bits on every
//  run (the grouping depends on n only)
void test_floats() {
  for (auto const n : SIZES) {
    std::vector<float> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = std::sin(0.001f * static_cast<float>(i)) + 0.25f;
    DeviceMem<float> d(h);
    auto const inf = std::numeric_limits<float>::infinity();
    MKN_KUL_TEST_CHECK(reduce(d, inf, reduce_op::Min{}) == *std::min_element(h.begin(), h.end()),
                       "min n=" + std::to_string(n));
    MKN_KUL_TEST_CHECK(reduce(d, -inf, reduce_op::Max{}) == *std::max_element(h.begin(), h.end()),
                       "max n=" + std::to_string(n));
    double abs_sum = 0, sum = 0;
    for (auto const v : h) sum += v, abs_sum += std::abs(v);
    auto const got = reduce(d, 0.f, reduce_op::Sum{});
    MKN_KUL_TEST_CHECK(std::abs(got - sum) <= 1e-5 * abs_sum + 1e-6,
                       "sum n=" + std::to_string(n) + " got " + std::to_string(got) + " expected " +
                           std::to_string(sum));
    MKN_KUL_TEST_CHECK(std::bit_cast<std::uint32_t>(reduce(d, 0.f, reduce_op::Sum{})) ==
                           std::bit_cast<std::uint32_t>(got),
                       "same bits n=" + std::to_string(n));
  }
}

// no values: init; init applied once (not once per thread or block)
void test_init() {
  DeviceMem<double> empty;
  MKN_KUL_TEST_CHECK(reduce(empty.data(), 0, 3.5, reduce_op::Sum{}) == 3.5);
  MKN_KUL_TEST_CHECK(reduce(empty.data(), 0, -1., reduce_op::Max{}) == -1.);
  std::vector<double> h(5000, 0.);
  DeviceMem<double> d(h);
  MKN_KUL_TEST_CHECK(reduce(d, 1., reduce_op::Sum{}) == 1.);
  MKN_KUL_TEST_CHECK(reduce(d, -2., reduce_op::Min{}) == -2.);
  MKN_KUL_TEST_CHECK(reduce(d, 2., reduce_op::Max{}) == 2.);
}

int main() {
  KOUT(NON) << __FILE__;
  test_integers();
  test_doubles_exact();
  test_floats();
  test_init();
}
