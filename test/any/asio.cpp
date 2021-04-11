
#include "kul/gpu.hpp"
#include "kul/gpu/asio.hpp"

#include  <algorithm>

static constexpr uint32_t BATCHES = 4;
static constexpr uint32_t NUM = 1024 * 1024 * BATCHES;
static constexpr uint32_t TP_BLOCK = 256;

struct A{
  std::uint32_t f0;
};

__global__ void single(A* a, int offset) {
  auto i = kul::gpu::asio::idx() + offset;
  a[i].f0 = a[i].f0 + 1;
}

uint32_t test_single(){
  kul::gpu::HostArray<A, NUM> a;
  for (uint32_t i = 0; i < NUM; ++i) a[i].f0 = i;
  kul::gpu::asio::Batch batch{BATCHES, a};
  kul::gpu::asio::Launcher{TP_BLOCK}(single, batch).async_back();
  for(std::size_t i = 0; i < batch.streams.size(); ++i){
    auto offset = i * batch.streamSize;
    auto copy_back = batch[i];
    for (uint32_t j = 0; j < batch.streamSize; ++j)
     if (copy_back[j].f0 != a[j + offset].f0 + 1)
       return 1;
  }
  return 0;
}

// struct B{
//   float f0;
// };

// __global__ void multiple(A* a, B* b) {
//   auto i = kul::gpu::idx();
//   a[i].f0 = b[i].f0 + 1;
// }
// uint32_t test_multiple(){
//   kul::gpu::HostArray<A, NUM> a;
//   kul::gpu::HostArray<B, NUM> b;
//   for (uint32_t i = 0; i < NUM; ++i) a[i].f0 = i;

//   kul::gpu::asio::Batch batch{NUM / 4, a, b};

//   kul::gpu::asio::Launcher{WIDTH, HEIGHT, TP_BLOCK, TP_BLOCK_Y}(multiple, batch);

//   // auto [copy_back] = dev();

//   // for (uint32_t i = 0; i < NUM; ++i)
//   //   if (copy_back[i].f0 != a[i].f0 + 1) return 1;

//   return 0;
// }

int main() {
  KOUT(NON) << __FILE__;
  return test_single(); //+ test_single();
}
