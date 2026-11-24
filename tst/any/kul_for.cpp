#include "mkn/gpu.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/range.hpp"
#include "mkn/kul/test.hpp"

#include <array>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <ranges>

MKN_KUL_TEST_DECLARE();

// mkn.kul's constexpr loop helpers called from device code
static constexpr std::uint32_t NUM = 1024, TPB = 256;  // 1D grid

template <typename T>
__global__ void kul_for(T* a) {
  auto const i = mkn::gpu::idx();

  // for_N_make_array: {i, i + 1, i + 2}
  auto const arr = mkn::kul::for_N_make_array<3>([&](auto ic) { return T(i + ic()); });

  // for_N_fold: every index as a pack
  T sum = mkn::kul::for_N_fold<3>([&](auto... ics) { return (arr[ics] + ...); });

  // for_N, no return
  mkn::kul::for_N<2>([&](auto ic) { sum += T(ic()); });

  // for_N_all through std::all_of
  if (!mkn::kul::for_N_all<3>([&](auto ic) { return arr[ic] >= T(i); })) sum = -1;

  for (auto const j : mkn::kul::range(4)) sum += T(j);

  a[i] = sum;
}

template <typename T>
void test() {
  mkn::gpu::DeviceMem<T> devA(NUM);
  mkn::gpu::GLauncher{NUM, 1, TPB, 1}(kul_for<T>, devA);
  auto a = devA();
  // 3i + 3 from the fold, + 1 from for_N, + 6 from range(4)
  MKN_KUL_TEST_CHECK(std::ranges::all_of(std::views::iota(0u, NUM),
                                         [&](auto const i) { return a[i] == T(3 * i + 10); }),
                     "a[i] == T(3 * i + 10)");
}

int main() {
  KOUT(NON) << __FILE__;
  test<float>();
  test<double>();
}
