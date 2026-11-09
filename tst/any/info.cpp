
#include "mkn/gpu.hpp"
#include "mkn/kul/log.hpp"

int main() {
  KOUT(NON) << __FILE__;
  mkn::gpu::prinfo();
  return 0;
}
