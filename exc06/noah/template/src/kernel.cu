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
reduction_Kernel(int numElements, float *dataIn, float *dataOut)
{

    extern __shared__ float sdata[];
    unsigned int tid   = threadIdx.x;
    unsigned int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Each thread computes a local sum over multiple elements
    float sum = 0.0f;
    for (int i = idx; i < numElements; i += stride) {
        sum += dataIn[i];
    }

    // Store local sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block (binary tree)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        dataOut[blockIdx.x] = sdata[0];
    }
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float *dataIn, float *dataOut)
{
    float* d_partial = nullptr;
    int numBlocks = gridSize.x;
    cudaMalloc(&d_partial, numBlocks * sizeof(float));

    reduction_Kernel<<<gridSize,
                       blockSize,
                       blockSize.x * sizeof(float)>>>(
        numElements, dataIn, d_partial);

    int threadsSecond = min(numBlocks, 1024);
    reduction_Kernel<<<1,
                       threadsSecond,
                       threadsSecond * sizeof(float)>>>(
        numBlocks, d_partial, dataOut);

    cudaFree(d_partial);
}

//
// Reduction Kernel using CUDA Thrust
//

void thrust_reduction_Wrapper(int numElements, float *dataIn, float *dataOut)
{
	thrust::device_ptr<float> in_ptr = thrust::device_pointer_cast(dataIn);
	thrust::device_ptr<float> out_ptr = thrust::device_pointer_cast(dataOut);

	*out_ptr = thrust::reduce(in_ptr, in_ptr + numElements, (float)0., thrust::plus<float>());
}
