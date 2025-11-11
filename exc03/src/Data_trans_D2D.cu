#include <stdio.h>
#include <cuda_runtime.h>
#include "chTimer.h"

int main()
{
    int data[] = {2, 8, 64, 512, 4096, 32768, 262144, 2097152, 16777216, 134217728, 1073741824};
    int iter[] = {2097152, 32768, 4096, 512, 64, 8, 2, 1, 1, 1};

    printf("\nMeasuring data transfer d2d (device to device)...\n"); fflush(stdout);

    for (int i = 0; i < 10; i++) {
        size_t dat = data[i];
        void *dmem1, *dmem2;
        cudaMalloc(&dmem1, dat);
        cudaMalloc(&dmem2, dat);

        chTimerTimestamp start, stop;
        chTimerGetTime(&start);

        for (int j = 0; j < iter[i]; j++) {
            cudaMemcpy(dmem2, dmem1, dat, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
        }

        chTimerGetTime(&stop);
        double microseconds = 1e6 * chTimerElapsedTime(&start, &stop);
        double usPerTransfer = microseconds / (float)iter[i];
        double datarate = 1e6 * dat / usPerTransfer;
        printf("%zu %.6f\n", dat, datarate);

        cudaFree(dmem1);
        cudaFree(dmem2);
    }

    return 0;
}
