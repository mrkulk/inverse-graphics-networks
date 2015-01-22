// File: test.cu
/* Usage:
GPU:
nvcc -m64 -arch=sm_20 -o test.so --shared -Xcompiler -fPIC test.cu
*/


#include <stdio.h>

__global__ void myk(void)
{
    printf("Hello from thread %d block %d\n", threadIdx.x, blockIdx.x);
}

extern "C"
void entry(void)
{
    myk<<<1,1>>>();
    printf("CUDA status: %d\n", cudaDeviceSynchronize());
}