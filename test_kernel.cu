
#include <stdio.h>
extern "C" {
#include "test_kernel.h"
}

__global__ void cuda_dot(double a, double *help)
{
   *help=2*a;
}

//kernel calling function
extern "C" 
void cuda_GMRESfunc(double a)
{
	double b;

	double *dev_a;
	double *res;

	cudaMemcpy(dev_a, &a, sizeof(double), cudaMemcpyHostToDevice );
	cuda_dot<<< 1, 1 >>>(*dev_a, res );
	cudaMemcpy(&b, res, sizeof(double), cudaMemcpyDeviceToHost );
}    


