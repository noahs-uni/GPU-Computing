#include <stdio.h>
#include <math.h>
#include "chTimer.h"

int
main()
{
    int data[] = {2, 8, 64, 512, 4096, 32768, 262144, 2097152, 16777216, 134217728, 1073741824};
    int iter[] = {2097152, 32768, 4096, 512, 64, 8, 2, 1, 1, 1};

    printf( "Measuring data transfer h2d using peageable memory... \n" ); fflush( stdout );


    for (int i = 0; i<10; i++){
        size_t dat = data[i];
        void *dmem, *hmem;
        cudaMalloc(&dmem, dat); // Allocate GPU memory
        hmem = malloc (dat);

        chTimerTimestamp start, stop;
        chTimerGetTime( &start );

        for ( int j = 0; j < iter[i]; j++ ) {
            cudaMemcpy(dmem, hmem, dat, cudaMemcpyHostToDevice);
        }

        cudaDeviceSynchronize();
        chTimerGetTime( &stop );
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerTransfer = microseconds / (float) iter[i];
        double datarate = 1e6 * dat / usPerTransfer;
        printf( "%zu %.6f %.6f\n", dat, usPerTransfer, datarate );

        cudaFree(dmem);
        free(hmem);

    }

    printf( "Measuring data transfer h2d using pinned memory... \n" ); fflush( stdout );


    for (int i = 0; i<10; i++){
        size_t dat = data[i];
        void *dmem, *hmem;

        cudaMalloc(&dmem, dat); // Allocate GPU memory
        cudaMallocHost (&hmem, dat);

        chTimerTimestamp start, stop;
        chTimerGetTime( &start );

        for ( int j = 0; j < iter[i]; j++ ) {
            cudaMemcpy(dmem, hmem, dat, cudaMemcpyHostToDevice);
        }

        cudaDeviceSynchronize();
        chTimerGetTime( &stop );
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerTransfer = microseconds / (float) iter[i];
        double datarate = 1e6 * dat / usPerTransfer;
        printf( "%zu %.6f %.6f\n", dat, usPerTransfer, datarate );

        cudaFree(dmem);
        cudaFreeHost(hmem);
    }

    return 0;
}
