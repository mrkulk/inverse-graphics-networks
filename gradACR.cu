// nvcc -m64 -shared -arch=sm_20 -o libgradACR.so  -Xcompiler -fPIC gradACR.cu

#include <stdio.h>

extern "C" {
#include "gradACR.h"
}
#include <cuda.h>

__global__ void test(void)
{
    //printf("Hello from thread %d block %d\n", threadIdx.x, blockIdx.x);
    //for(int i=0; i<1000; i++) {
    //	int a = 1;
    // 	for(int j=1;j<10000;j++) {
    //		int b = a*10;
    //	}
    //}
}

__device__ double kernel_getTemplateValue(int bsize,int tdim, int bid, double *ktemplate, double template_x, double template_y) 
{

	double res=0;
	double template_x_size = tdim + 1; 
	double template_y_size = tdim + 1; 
	int output_x =  floor(template_x + template_x_size/2) - 1; //because we added +1 in beginning
	int output_y = floor(template_y + template_y_size/2) - 1; //because we added +1 in beginning

	if (output_x < 1 || output_x > tdim || output_y < 1 || output_y > tdim) {
		res = 0.0;
	}
	else {
		res = ktemplate[bid*tdim*tdim + tdim*output_x + output_y];
	}
	return res;
}


__device__ double atomicAdd_double(double* address, double val)
{
	unsigned long long int* address_as_ull =
	(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
	assumed = old;
	old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
	__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void getgradient(int imwidth, int tdim, int bsize, double *cuda_output, double *cuda_pose, double *cuda_template, double *cuda_gradOutput, double *cuda_gradTemplate, double *cuda_gradPose)
{
	int i,j;

	//printf("block: (%d %d %d) || grid: (%d %d %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
	unsigned int bid = threadIdx.z; //index of image in batch
	unsigned int output_x = blockIdx.x + 1; //critical -- adding 1 due to lua terminology
	unsigned int output_y = blockIdx.y + 1; //critical -- adding 1 due to lua terminology


	unsigned int output_coords[3];
	output_coords[0]=output_x; output_coords[1]=output_y; output_coords[2]=1;

	double template_coords[3];
	double pose[3][3];
	for(i=0;i<9;i++) {
		pose[int(i/3)][i%3] = cuda_pose[bid*9 + i];
	}

	//matrix mul
	for(i=0;i<3;i++){
		template_coords[i]=0;
		for(j=0;j<3;j++) {
			template_coords[i] += pose[i][j]*output_coords[j];
		}
	}


	double template_x = template_coords[0] - 0.5;
	double template_y = template_coords[1] - 0.5;

	
	float x_high_coeff = fmod(template_x , 1); 
	float y_high_coeff = fmod(template_y ,1); 	
	
	double x_low_coeff = -x_high_coeff + 1;
	double y_low_coeff = -y_high_coeff + 1;

	int x_low = floor(template_x);
	int x_high = x_low + 1;
	int y_low = floor(template_y);
	int y_high = y_low + 1;

	///////////// Pose Gradient Initial /////////////


	double template_val_xhigh_yhigh = kernel_getTemplateValue( bsize, tdim, bid, cuda_template, x_high, y_high);
	double template_val_xhigh_ylow = kernel_getTemplateValue( bsize, tdim, bid, cuda_template, x_high, y_low);
	double template_val_xlow_ylow = kernel_getTemplateValue( bsize, tdim, bid, cuda_template, x_low, y_low);
	double template_val_xlow_yhigh = kernel_getTemplateValue( bsize, tdim, bid, cuda_template, x_low, y_high);

	double pose_1_1, pose_1_2, pose_1_3, pose_2_1, pose_2_2, pose_2_3;
	pose_1_1 = pose[0][0]; pose_1_2 = pose[0][1]; pose_1_3 = pose[0][2];
	pose_2_1 = pose[1][0]; pose_2_2 = pose[1][1]; pose_2_3 = pose[1][2];

	double cache1,cache2, cache3,cache4, cache5, cache6, cache7;
	double cache8, cache9, cache10, cache11, cache12, cache13, cache14;

	cache1 = pose_2_3 - y_low + pose_2_1*output_x + pose_2_2*output_y;
	cache2 = pose_2_3 - y_high + pose_2_1*output_x + pose_2_2*output_y;
	cache3 = pose_1_3 - x_low + pose_1_1*output_x + pose_1_2*output_y;
	cache4 = pose_1_3 - x_high + pose_1_1*output_x + pose_1_2*output_y;

	cache5 = template_val_xhigh_yhigh * cache3;
	cache6 = template_val_xlow_yhigh * cache4;
	cache7 = template_val_xhigh_ylow * cache3;
	cache8 = template_val_xlow_ylow * cache4;

	//need to subtract 1 from output_x and output_y as we added in the beginning and cant keep while indexing
	double cache_gradOutput_outputx_outputy = cuda_gradOutput[bid*imwidth*imwidth + imwidth*(output_x-1) + (output_y-1)];

	cache9 = cache_gradOutput_outputx_outputy * (cache5-cache6);
	cache10 = cache7 - cache8;

	cache11 = (template_val_xhigh_ylow - template_val_xlow_ylow)*cache2;
	cache12 = cache_gradOutput_outputx_outputy*( (template_val_xhigh_yhigh - template_val_xlow_yhigh)*cache1 );

	cache13 = cache12 - cache11;
	cache14 = cache9 - cache10;
	
	///////////// Template Gradient Initial /////////
	double x_vec[2], y_vec[2];
	x_vec[0]=x_low_coeff; x_vec[1]=x_high_coeff;
	y_vec[0]=y_low_coeff; y_vec[1]=y_high_coeff;

	double dOutdPose[2][2]; //outer-product
	dOutdPose[0][0] = x_vec[0]*y_vec[0];
	dOutdPose[0][1] = x_vec[0]*y_vec[1];
	dOutdPose[1][0] = x_vec[1]*y_vec[0];
	dOutdPose[1][1] = x_vec[1]*y_vec[1];

	
	////////////////////// accumulate gradient ////////////////
	///////////// using atomics to avoid race condition ///////
	///////////// subtracting -1 from x_low ...etc as we added +1 in beginning ///////
	if (x_low >= 1 && x_low <= tdim && y_low >= 1 && y_low <= tdim) {
		atomicAdd_double(&(cuda_gradTemplate[bid*tdim*tdim + tdim*(x_low-1) + (y_low-1)]), dOutdPose[0][0]);
	}

	if (x_low >= 1 && x_low <= tdim && y_high >= 1 && y_high <= tdim) { 
		atomicAdd_double(&(cuda_gradTemplate[bid*tdim*tdim + tdim*(x_low-1) + (y_high-1)]), dOutdPose[0][1]);
	}
	if (x_high >= 1 && x_high <= tdim && y_low >= 1 && y_low <= tdim) {
		atomicAdd_double(&(cuda_gradTemplate[bid*tdim*tdim + tdim*(x_high-1) + (y_low-1)]), dOutdPose[1][0]);
	}
	
	if (x_high >= 1 && x_high <= tdim && y_high >= 1 && y_high <= tdim){ 
		atomicAdd_double(&(cuda_gradTemplate[bid*tdim*tdim + tdim*(x_high-1) + (y_high-1)]), dOutdPose[1][1]);
	}

	atomicAdd_double(&(cuda_gradPose[bid*9]), 	  cache13*output_x);
	atomicAdd_double(&(cuda_gradPose[bid*9 + 1]), cache13*output_y);
	atomicAdd_double(&(cuda_gradPose[bid*9 + 2]), cache12 - cache11);
	atomicAdd_double(&(cuda_gradPose[bid*9 + 3]), cache14 * output_x);
	atomicAdd_double(&(cuda_gradPose[bid*9 + 4]), cache14 * output_y);
	atomicAdd_double(&(cuda_gradPose[bid*9 + 5]), (cache_gradOutput_outputx_outputy*cache5)-cache6-cache7+cache8);	
	
	//printf("%d %d %f\n",output_x, output_y ,cache14);
	/*if (output_x == 3 && output_y==4) {
		printf("GPU: %f \n", cache14);
		//for (i=0;i<11;i++) {
		//	printf("\n");
		//	for (j=0;j<11;j++) {
		//		printf("%.4f ", cuda_template[i*11 + j]);
		//	}	
		//}
	}*/
}		


extern "C" void get_gradACR_gradient(int imwidth, int tdim, int bsize, double *output, double *pose, 
						double *_template, double *gradOutput, double *gradTemplate, double *gradPose)
{	
	int output_size = sizeof(double) * bsize * imwidth * imwidth ;
	int pose_size =  sizeof(double) * bsize * 3 * 3;
	int template_size = sizeof(double) * bsize * tdim * tdim ;
	int gradOutput_size = sizeof(double) * bsize * imwidth * imwidth;
	int gradTemplate_size = sizeof(double) * bsize * tdim * tdim;
	int gradPose_size = sizeof(double) * bsize * 3 * 3;

 	double *cuda_output, *cuda_pose, *cuda_template, 
 		   *cuda_gradOutput, *cuda_gradTemplate, *cuda_gradPose; 

	cudaMalloc((void**)&cuda_output, output_size);
	cudaMemcpy(cuda_output, output, output_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_pose, pose_size);
	cudaMemcpy(cuda_pose, pose, pose_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_template, template_size);
	cudaMemcpy(cuda_template, _template, template_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_gradOutput, gradOutput_size);
	cudaMemcpy(cuda_gradOutput, gradOutput, gradOutput_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_gradTemplate, gradTemplate_size);
	cudaMemcpy(cuda_gradTemplate, gradTemplate, gradTemplate_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_gradPose, gradPose_size);
	cudaMemcpy(cuda_gradPose, gradPose, gradPose_size, cudaMemcpyHostToDevice);

	cudaMemset((void **)&cuda_gradTemplate, 0, gradTemplate_size);
	cudaMemset((void **)&cuda_gradPose, 0, gradPose_size);
	
	//setup GPU grid and block structure
	dim3 grid; grid.x=32; grid.y = 32;
	//dim3 grid; grid.x=3; grid.y = 3;

	dim3 block; block.x=1; block.y=1; block.z=bsize;
	//dim3 block; block.x=1; block.y=1; block.z=2;

	getgradient<<<grid,block>>>(imwidth, tdim,  bsize, cuda_output, cuda_pose, cuda_template, cuda_gradOutput, cuda_gradTemplate, cuda_gradPose);

    cudaMemcpy(gradTemplate, cuda_gradTemplate, gradTemplate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gradPose, cuda_gradPose, gradPose_size, cudaMemcpyDeviceToHost);
    
    printf("CUDA status: %d\n", cudaDeviceSynchronize());
}


/*
extern "C" void get_gradACR_gradient(int imwidth, int tdim, int bsize, double *cuda_output, double *cuda_pose, 
						double *cuda_template, double *cuda_gradOutput, double *cuda_gradTemplate, double *cuda_gradPose)
{	
	

	//setup GPU grid and block structure
	//dim3 grid; grid.x=32; grid.y = 32;
	dim3 grid; grid.x=3; grid.y = 3;

	//dim3 block; block.x=1; block.y=1; block.z=bsize;
	dim3 block; block.x=1; block.y=1; block.z=2;

    getgradient<<<grid,block>>>(imwidth, tdim,  bsize, cuda_output, cuda_pose, cuda_template, cuda_gradOutput, cuda_gradTemplate, cuda_gradPose);
    printf("CUDA status: %d\n", cudaDeviceSynchronize());
}*/
