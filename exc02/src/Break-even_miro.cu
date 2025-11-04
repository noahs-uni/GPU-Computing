#include <stdio.h>

#include "chTimer.h"

__device__ unsigned long long d_cycles;

__global__ void busyWait(unsigned long long wait_cycles, bool record)
{
    unsigned long long start = clock64();
    while (clock64() - start <= wait_cycles) {
    }
    if (record) {
        d_cycles = clock64() - start;
    }
}

int
main()
{
    const int N = 100;
    const unsigned long long start_cycles = 1;
    const unsigned long long step = 100;  
    unsigned long long *wait_array = (unsigned long long*) malloc(N * sizeof(unsigned long long));

    for (int i = 0; i < N; i++) {
        wait_array[i] = start_cycles + i * step;
    }

    bool record = false;
    const int cIterations = 100000;
    printf( "Measuring asynchronous launch time... " ); fflush( stdout );

    for (int j = 0; j < N; j++) {
        chTimerTimestamp start, stop;
        unsigned long long wait_cycles = wait_array[j];
        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) {
            busyWait<<<1,1>>>(wait_cycles, record);
        }
        cudaDeviceSynchronize();
        chTimerGetTime( &stop );

        {
            double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
            double usPerLaunch = microseconds / (float) cIterations;

            printf( "%llu %.6f\n", wait_cycles, usPerLaunch );
        }
    }

    return 0;
}