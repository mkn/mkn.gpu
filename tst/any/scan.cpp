#include "mkn/gpu.hpp"
#include "mkn/gpu/scan.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/test.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

MKN_KUL_TEST_DECLARE();

using namespace mkn::gpu;

template <typename T>
using ManagedVector = std::vector<T, ManagedAllocator<T>>;

// around a tile (2048), the one block limit (8 tiles) and beyond
static std::size_t constexpr SIZES[] = {1, 2, 255, 2047, 2048, 2049, 16384, 16385, 100003, 3000017};

template <typename T>
void check(std::size_t const n, bool const in_place) {
  ManagedVector<T> in(n), out(n);
  std::vector<T> expect(n);
  for (std::size_t i = 0; i < n; ++i) in[i] = static_cast<T>((i * 2654435761u) % 7u);
  std::exclusive_scan(in.begin(), in.end(), expect.begin(), T{0});
  T const sum = std::accumulate(in.begin(), in.end(), T{0});
  T* const dst = in_place ? in.data() : out.data();
  T const total = exclusive_scan(in.data(), dst, n);
  bool ok = total == sum;
  for (std::size_t i = 0; i < n; ++i) ok = ok and dst[i] == expect[i];
  MKN_KUL_TEST_CHECK(ok, "n=" + std::to_string(n) + (in_place ? " in place" : ""));
}

void test_sizes() {
  for (auto const n : SIZES) {
    check<std::uint32_t>(n, false);
    check<std::uint32_t>(n, true);
    check<std::int64_t>(n, false);
  }
  std::uint32_t const* none = nullptr;
  MKN_KUL_TEST_CHECK(exclusive_scan(none, static_cast<std::uint32_t*>(nullptr), 0) == 0, "n=0");
}

int main() {
  KOUT(NON) << __FILE__;
  test_sizes();
}
