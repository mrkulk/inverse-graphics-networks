// nvcc -arch=sm_20 -L. -lgradACR testgradACR.c

#include "gradACR.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
 
int main(){
	//args:(mode, start_x, start_y, endhere_x, endhere_y, output, pose, bsize, template, gradOutput, _gradTemplate, _gradPose)
  	/*start_x = 1;
  	start_y = 1;
  	endhere_x = 1;
  	endhere_y = 1;
  	output = */
  	int bsize = 30;
  	int imwidth = 32;
  	int tdim = 11; //template size 
  	int i;

	double *output = 0; double *cuda_output;
	double *pose = 0; double *cuda_pose; 
	double *template = 0; double *cuda_template;
	double *gradOutput = 0; double *cuda_gradOutput;
	double *gradTemplate = 0; double *cuda_gradTemplate;
	double *gradPose = 0; double *cuda_gradPose;

	int output_size = sizeof(double) * bsize * imwidth * imwidth ;
	int pose_size =  sizeof(double) * bsize * 3 * 3;
	int template_size = sizeof(double) * bsize * tdim * tdim ;
	int gradOutput_size = sizeof(double) * bsize * imwidth * imwidth;
	int gradTemplate_size = sizeof(double) * bsize * tdim * tdim;
	int gradPose_size = sizeof(double) * bsize * 3 * 3;

	output = (double * )malloc(output_size);
	for( i=0; i < output_size/sizeof(double); i++) { output[i] = rand()%5; }

	pose = 	(int * )malloc(pose_size );
	for( i=0; i < pose_size/sizeof(double); i++) { 
		pose[i] = 0.6*(rand()%5); 
	}
	
	template=(double * )malloc(template_size);
	for( i=0; i < template_size/sizeof(double); i++) { 
		template[i] = rand()%5; 
	}

	gradOutput = (double * )malloc(gradOutput_size );
	for( i=0; i < gradOutput_size/sizeof(double); i++) { 
		gradOutput[i] = rand()%10; 
	}

	gradTemplate = (double * )malloc(gradTemplate_size );
	gradPose = (double * )malloc(gradPose_size );

	cudaMalloc((void**)&cuda_output, output_size);
	cudaMemcpy(cuda_output, output, output_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_pose, pose_size);
	cudaMemcpy(cuda_pose, pose, pose_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_template, template_size);
	cudaMemcpy(cuda_template, template, template_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_gradOutput, gradOutput_size);
	cudaMemcpy(cuda_gradOutput, gradOutput, gradOutput_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_gradTemplate, gradTemplate_size);
	cudaMemcpy(cuda_gradTemplate, gradTemplate, gradTemplate_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_gradPose, gradPose_size);
	cudaMemcpy(cuda_gradPose, gradPose, gradPose_size, cudaMemcpyHostToDevice);

	cudaMemset((void **)&cuda_gradTemplate, 0, gradTemplate_size);
	cudaMemset((void **)&cuda_gradPose, 0, gradPose_size);

	get_gradACR_gradient(imwidth, tdim ,bsize, cuda_output, cuda_pose, cuda_template, cuda_gradOutput, cuda_gradTemplate, cuda_gradPose);
	return 0;
}