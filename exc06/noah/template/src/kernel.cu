/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : 01
 *
 *                   File : kernel.cu
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

//
// Reduction_Kernel
//
__global__ void
reduction_Kernel(int numElements, float* dataIn, float* dataOut)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (elementId < numElements)
	{
		/*TODO Kernel Code*/
		/* add additional for loops if requrired, e.g., if the max. grid size is not sufficient for the problem size */
	}
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	reduction_Kernel<<< gridSize, blockSize>>>(numElements, dataIn, dataOut);
}

//
// Reduction Kernel using CUDA Thrust
//

void thrust_reduction_Wrapper(int numElements, float* dataIn, float* dataOut) {
	thrust::device_ptr<float> in_ptr = thrust::device_pointer_cast(dataIn);
	thrust::device_ptr<float> out_ptr = thrust::device_pointer_cast(dataOut);
	
	*out_ptr = thrust::reduce(in_ptr, in_ptr + numElements, (float) 0., thrust::plus<float>());	
}
