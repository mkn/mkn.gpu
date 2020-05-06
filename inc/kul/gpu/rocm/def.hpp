

#ifndef _KUL_GPU_ROCM_DEF_HPP_
#define _KUL_GPU_ROCM_DEF_HPP_

namespace kul::gpu::hip {

__device__ uint32_t idx() {
  uint32_t width = hipGridDim_x * hipBlockDim_x;
  uint32_t height = hipGridDim_y * hipBlockDim_y;

  uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  uint32_t z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  return x + (y * width) + (z * width * height); // max 4294967296
}

}  // namespace kul::gpu::hip

/* // make test of
__global__ void xyz(uint32_t* in)
{
    auto i = kul::gpu::hip::idx();
    in[i]  = i;
}
int main(int argc, char** argv)
{
    std::vector<uint32_t> data(X * Y * Z, 123);
    kul::gpu::DeviceMem<uint32_t> gata{data};

    kul::gpu::Launcher{X, Y, Z, TPB_X, TPB_Y, TPB_Z}(xyz, gata.p);

    auto host = gata();

    KLOG(NON) << host[10];
    KLOG(NON) << host[100];
    KLOG(NON) << host[1000];
    KLOG(NON) << host[10000];
    KLOG(NON) << host[100000];
    KLOG(NON) << host[1000000];
    KLOG(NON) << host[2000000];
    KLOG(NON) << host.back();
    KLOG(NON) << host.size();

    return 0;
}*/

#endif /*_KUL_GPU_ROCM_DEF_HPP_*/
