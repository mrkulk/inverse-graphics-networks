// nvcc -m64 -shared -arch=sm_20 -o libtestkernel.so  -Xcompiler -fPIC test_kernel.cu

#include <stdio.h>

extern "C" {
#include "test_kernel.h"
}

__global__ void myk(void)
{
    printf("Hello from thread %d block %d\n", threadIdx.x, blockIdx.x);
}


extern "C" void entry(void)
{
    myk<<<1,1>>>();
    printf("CUDA status: %d\n", cudaDeviceSynchronize());
}