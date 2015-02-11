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

__device__ double kernel_getTemplateValue(int bsize,int tdim, int bid, double *ktemplate, int template_x, int template_y) 
{

	double res=0;
	if (template_x < 1 || template_x > tdim || template_y < 1 || template_y > tdim) {
		res = 0.0;
	}
	else {
		res = ktemplate[bid*tdim*tdim + tdim*(template_x-1) + (template_y-1)];
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


__global__ void getgradient(int imwidth, int tdim, int bsize, double *cuda_output, double *cuda_pose, double *cuda_template, double *cuda_gradOutput, double *cuda_gradTemplate, double *cuda_gradPose, double *cuda_intensity)
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

	double intensity = cuda_intensity[bid]; 

	double template_x = template_coords[0];
	double template_y = template_coords[1];

	double template_x_after_offset = template_x + (double(tdim) / 2.0);
    double template_y_after_offset = template_y + (double(tdim) / 2.0);


	int x_low = floor(template_x_after_offset);
	int x_high = x_low + 1;
	int y_low = floor(template_y_after_offset);
	int y_high = y_low + 1;

	//need to subtract 1 from output_x and output_y as we added in the beginning and cant keep while indexing
	double cache_gradOutput_outputx_outputy = cuda_gradOutput[bid*imwidth*imwidth + imwidth*(output_x-1) + (output_y-1)];

	///////////// Template Gradient Initial /////////
	///////////// subtracting -1 from x_low ...etc as we added +1 in beginning ///////
	///////////// using atomics to avoid race condition ///////

	int ratio_xy = 1;
	if (x_low >= 1 && x_low <= tdim && y_low >= 1 && y_low <= tdim ) {
	    double cur_gt_xlow_ylow = ( (((template_x_after_offset-x_high)*(template_y_after_offset-y_high))/ ratio_xy )
	                                                  * cache_gradOutput_outputx_outputy );
	    atomicAdd_double(&(cuda_gradTemplate[bid*tdim*tdim + tdim*(x_low-1) + (y_low-1)]), cur_gt_xlow_ylow);
	}

	if (x_low >= 1 && x_low <= tdim && y_high >= 1 && y_high <= tdim) {
	  double cur_gt_xlow_yhigh = ( -(((template_x_after_offset-x_high)*(template_y_after_offset-y_low))/ ratio_xy ) *cache_gradOutput_outputx_outputy );
	  atomicAdd_double(&(cuda_gradTemplate[bid*tdim*tdim + tdim*(x_low-1) + (y_high-1)]), cur_gt_xlow_yhigh);
	}

	if (x_high >= 1 && x_high <= tdim && y_low >= 1 && y_low <= tdim) {
	  double cur_gt_xhigh_ylow = ( -(((template_x_after_offset-x_low)*(template_y_after_offset-y_high))/ ratio_xy ) * cache_gradOutput_outputx_outputy);
	  atomicAdd_double(&(cuda_gradTemplate[bid*tdim*tdim + tdim*(x_high-1) + (y_low-1)]), cur_gt_xhigh_ylow);
	}

	if (x_high >= 1 && x_high <= tdim && y_high >= 1 && y_high <= tdim) {
	  double cur_gt_xhigh_yhigh = ( (((template_x_after_offset-x_low)*(template_y_after_offset-y_low))/ ratio_xy ) * cache_gradOutput_outputx_outputy);
   	  atomicAdd_double(&(cuda_gradTemplate[bid*tdim*tdim + tdim*(x_high-1) + (y_high-1)]), cur_gt_xhigh_yhigh);
	}


	///////////// Pose Gradient Initial /////////////
	double template_val_xhigh_yhigh = intensity* kernel_getTemplateValue( bsize, tdim, bid, cuda_template, x_high, y_high);
	double template_val_xhigh_ylow = intensity*kernel_getTemplateValue( bsize, tdim, bid, cuda_template, x_high, y_low);
	double template_val_xlow_ylow = intensity*kernel_getTemplateValue( bsize, tdim, bid, cuda_template, x_low, y_low);
	double template_val_xlow_yhigh = intensity*kernel_getTemplateValue( bsize, tdim, bid, cuda_template, x_low, y_high);

	double cache1 = (template_y_after_offset - y_low);  
	double cache2 = (template_y_after_offset - y_high); 
	double cache3 = (template_x_after_offset - x_low);
	double cache4 = (template_x_after_offset - x_high);

	double cache5 = template_val_xlow_ylow * cache2;
	double cache6 = template_val_xlow_yhigh * cache1;
	double cache7 = template_val_xhigh_ylow * cache2;
	double cache8 = template_val_xhigh_yhigh * cache1;

	double cache9 = (cache5 - cache6 - cache7 + cache8);

	double cache10 = template_val_xlow_ylow * cache4;
	double cache11 = template_val_xlow_yhigh * cache4;
	double cache12 = template_val_xhigh_ylow * cache3;
	double cache13 = template_val_xhigh_yhigh * cache3;

	double cache14 = (cache10 - cache11 - cache12 + cache13);


	atomicAdd_double(&(cuda_gradPose[bid*9]), cache9 * double(output_x) * cache_gradOutput_outputx_outputy );

	atomicAdd_double(&(cuda_gradPose[bid*9 + 1]), cache9 * double(output_y) * cache_gradOutput_outputx_outputy );

	atomicAdd_double(&(cuda_gradPose[bid*9 + 2]), cache9 * cache_gradOutput_outputx_outputy );

	atomicAdd_double(&(cuda_gradPose[bid*9 + 3]), cache14 * double(output_x) * cache_gradOutput_outputx_outputy );

	atomicAdd_double(&(cuda_gradPose[bid*9 + 4]), cache14 * double(output_y) * cache_gradOutput_outputx_outputy );

	atomicAdd_double(&(cuda_gradPose[bid*9 + 5]), cache14 * cache_gradOutput_outputx_outputy );	
	
	
	//cuda_gradPose[bid*9] = 66;
	//cuda_gradPose[bid*9+1] = 2;
	//cuda_gradPose[bid*9+2] = 3;
	//cuda_gradPose[bid*9+3] = 4;
	//cuda_gradPose[bid*9+4] = 5;
	//cuda_gradPose[bid*9+5] = 6;

	//double tmp = cache9 * double(output_x) * cache_gradOutput_outputx_outputy;
	//if (tmp != 0 && bid==0) printf("bid: %d | %d %d %f \n", bid, output_x, output_y,tmp);

	//printf("%d %d\n", output_x, output_y);

	//printf("%d %d %f\n",output_x, output_y ,cache_gradOutput_outputx_outputy);
	//if (output_x == 3 && output_y==4) {
	//	printf("\n\nGPU: %f %d %d\n", template_val_xlow_ylow, x_low, y_low );
		//printf("GPU: %f %f\n", template_x_after_offset, template_y_after_offset);
		//for (i=0;i<11;i++) {
		//	printf("\n");
		//	for (j=0;j<11;j++) {
		//		printf("%.4f ", cuda_template[i*11 + j]);
		//	}	
		//}
	//}
}		


extern "C" void get_gradACR_gradient(int imwidth, int tdim, int bsize, double *output, double *pose, 
						double *_template, double *gradOutput, double *gradTemplate, double *gradPose, double *intensity)
{	
	int output_size = sizeof(double) * bsize * imwidth * imwidth ;
	int pose_size =  sizeof(double) * bsize * 3 * 3;
	int template_size = sizeof(double) * bsize * tdim * tdim ;
	int gradOutput_size = sizeof(double) * bsize * imwidth * imwidth;
	int gradTemplate_size = sizeof(double) * bsize * tdim * tdim;
	int gradPose_size = sizeof(double) * bsize * 3 * 3;
	int intensity_size = sizeof(double) * bsize;

 	double *cuda_output, *cuda_pose, *cuda_template, 
 		   *cuda_gradOutput, *cuda_gradTemplate, *cuda_gradPose, *cuda_intensity; 

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

	cudaMalloc((void**)&cuda_intensity, intensity_size);
	cudaMemcpy(cuda_intensity, intensity, intensity_size, cudaMemcpyHostToDevice);

	cudaMemset((void **)&cuda_gradTemplate, 0, gradTemplate_size);
	cudaMemset((void **)&cuda_gradPose, 0, gradPose_size);
	
	//setup GPU grid and block structure
	dim3 grid; grid.x=imwidth; grid.y = imwidth;
	//dim3 grid; grid.x=3; grid.y = 3;

	dim3 block; block.x=1; block.y=1; block.z=bsize;
	//dim3 block; block.x=1; block.y=1; block.z=2;

	getgradient<<<grid,block>>>(imwidth, tdim,  bsize, cuda_output, cuda_pose, cuda_template, cuda_gradOutput, cuda_gradTemplate, cuda_gradPose, cuda_intensity);

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
