/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : gpu01
 *
 *                            File : kernel.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

//
// Kernels
//

__global__ void 
globalMemCoalescedKernel(int *dmem1, int *dmem2, size_t dataSize)
{
    int numElements = dataSize / sizeof(int);
    
    int totalThreads = gridDim.x * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numElements) {
        for (int i = idx; i < numElements; i += totalThreads) {
            dmem1[i] = dmem2[i];
        }
    }
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *dmem1, int *dmem2, size_t dataSize) {
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 >>>(dmem1, dmem2, dataSize);
}

__global__ void 
globalMemStrideKernel(int *dmem1, int *dmem2, int stride)
{
    // Each thread accesses exactly one element with strided access pattern
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int elementIdx = idx * stride;
    
    // Access one element: dmem1[elementIdx] = dmem2[elementIdx]
    dmem1[elementIdx] = dmem2[elementIdx];
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *dmem1, int *dmem2, int stride) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 >>>(dmem1, dmem2, stride);
}

__global__ void 
globalMemOffsetKernel(int *dmem1, int *dmem2, int offset)
{
    // Each thread accesses exactly one element with offset access pattern
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int elementIdx = idx + offset;
    
    // Access one element: dmem1[elementIdx] = dmem2[elementIdx]
    dmem1[elementIdx] = dmem2[elementIdx];
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *dmem1, int *dmem2, int offset) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 >>>(dmem1, dmem2, offset);
}

