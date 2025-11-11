#include <stdio.h>
#include <math.h>
#include "chTimer.h"

int
main()
{
    size_t data[] = {2, 8, 64, 512, 4096, 32768, 262144, 2097152, 16777216, 134217728, 1073741824};
    size_t iter[] = {2097152, 32768, 4096, 512, 64, 8, 2, 1, 1, 1};

    printf( "Measuring device-to-device transfer bandwidth...\n" ); fflush( stdout );

    for (size_t i = 0; i < 10; ++i){
        size_t dat = data[i];
        void *dmem1, *dmem2;
        cudaMalloc(&dmem1, dat); // Allocate GPU memory
        cudaMalloc(&dmem2, dat); // Allocate GPU memory

        chTimerTimestamp start, stop;
        chTimerGetTime( &start );

        for ( size_t j = 0; j < iter[i]; j++ ) {
            cudaMemcpy(dmem1, dmem2, dat, cudaMemcpyDeviceToDevice);
        }

        cudaDeviceSynchronize();
        chTimerGetTime( &stop );
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerTransfer = microseconds / (float) iter[i];
        double datarate = 1e6 * dat / usPerTransfer;
        printf( "%zu %.6f %.6f\n", dat, usPerTransfer, datarate );

        cudaFree(dmem1);
        cudaFree(dmem2);

    }

    return 0;
}
