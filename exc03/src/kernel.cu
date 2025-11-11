/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

//
// Kernels
//

__global__ void 
globalMemCoalescedKernel(/*TODO Parameters*/)
{
    /*TODO Kernel Code*/
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

__global__ void 
globalMemStrideKernel(/*TODO Parameters*/)
{
    /*TODO Kernel Code*/
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

__global__ void 
globalMemOffsetKernel(/*TODO Parameters*/)
{
    /*TODO Kernel Code*/
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

