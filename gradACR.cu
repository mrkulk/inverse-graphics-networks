// nvcc -m64 -shared -arch=sm_20 -o libgradACR.so  -Xcompiler -fPIC gradACR.cu

#include <stdio.h>

extern "C" {
#include "gradACR.h"
}

__global__ void myk(void)
{
    printf("Hello from thread %d block %d\n", threadIdx.x, blockIdx.x);
}


extern "C" void get_gradACR_gradient(void)
{
    myk<<<32,32>>>();
    printf("CUDA status: %d\n", cudaDeviceSynchronize());
}