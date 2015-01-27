// nvcc -L. -lgradACR testgradACR.c

#include "gradACR.h"
#include <stdio.h>
#include <stdlib.h>

int main(){
	//args:(mode, start_x, start_y, endhere_x, endhere_y, output, pose, bsize, template, gradOutput, _gradTemplate, _gradPose)
  	/*start_x = 1;
  	start_y = 1;
  	endhere_x = 1;
  	endhere_y = 1;
  	output = */
  	int bsize = 30;

	double *output = 0; double *cuda_output;
	double *pose = 0;
	double *template = 0;
	double *gradOutput = 0;
	double *gradTemplate = 0;
	double *gradPose = 0;

	output = (double * )malloc(sizeof(double) * 30 * 32 * 32 );
	pose = 	 (double * )malloc(sizeof(double) * 30 * 3 * 3 );
	template=(double * )malloc(sizeof(double) * 30 * 11 * 11 );
	gradOutput = (double * )malloc(sizeof(double) * 30 * 32 * 32 );
	gradTemplate = (double * )malloc(sizeof(double) * 30 * 11 * 11 );
	gradPose = (double * )malloc(sizeof(double) * 30 * 3 * 3 );

	cudaMalloc((void**)&cuda_output, sizeof(double) * 30 * 32 * 32);

	get_gradACR_gradient();
	return 0;
}