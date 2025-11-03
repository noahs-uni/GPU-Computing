#include <stdio.h>

#include "chTimer.h"

__device__ unsigned long long d_writeback;

__global__
void
NullKernel()
{
}

__global__ void BusywaitKernel(unsigned long long wait_cycles, int write)
{
    unsigned long long start = clock64();
    unsigned long long stop = start;
    while ((stop - start) < wait_cycles) {
        stop = clock64();
    }
    if (write) {
        d_writeback = stop - start;
    }
}

int main()
{
    const int iterations = 10000;

    // initialize GPU context
    NullKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    
    chTimerTimestamp start, stop;
    // baseline measurement
    chTimerGetTime(&start); 
    for (int i = 0; i < iterations; i++) {
        NullKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
    chTimerGetTime(&stop);
    double baseline_time = chTimerElapsedTime(&start, &stop) / iterations;

    // Launch kernel with busy-waiting
    double busywait_time;
    double busywait_time_tot = 0;
    unsigned long long wait_cycles = 0;
    unsigned long long wait_cycles_tot = 0;
    for (int i = 0; i < iterations; i++) {
        wait_cycles = 0;
        do {
            wait_cycles += 100;
            chTimerGetTime(&start);     
            BusywaitKernel<<<1, 1>>>(wait_cycles, 0);
            cudaDeviceSynchronize();
            chTimerGetTime(&stop);
            busywait_time = chTimerElapsedTime(&start, &stop);
        } while (busywait_time < 2 * baseline_time);
        wait_cycles_tot += wait_cycles;
        busywait_time_tot += busywait_time;
    }

    cudaDeviceSynchronize();
    printf("Avg busy-wait cycles: %llu\n", wait_cycles_tot / iterations);
    printf("Baseline launch time: %f us\n", baseline_time * 1e6);
    printf("Avg busy-wait time: %f us\n", busywait_time_tot * 1e6 / iterations);

    return 0;
}