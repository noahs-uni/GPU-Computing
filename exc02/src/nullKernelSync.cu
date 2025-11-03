/*
 *
 * nullKernelAsync.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>

#include "chTimer.h"

__global__
void
NullKernel()
{
}

int
main()
{
    int gridSize[] = {1, 2, 4, 8, 32,  128, 512, 2048, 8192, 16384};
    int blockSize[] = {1, 2, 4, 8, 16, 32, 64, 128, 512, 1024};

    // initialize GPU context
    NullKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    const int cIterations = 100000;

    chTimerTimestamp startG[10], stopG[10], startB[10], stopB[10], startM[10], stopM[10];

    // iterate along grid sizes
    for (int g = 0; g < 10; g++) {
        chTimerGetTime( &startG[g] );
        for ( int i = 0; i < cIterations; i++ ) {
            NullKernel<<<gridSize[g],1>>>();
            cudaDeviceSynchronize();
        }
        chTimerGetTime( &stopG[g] );
    }

    // iterate along block sizes
    for (int b = 0; b < 10; b++) {
        chTimerGetTime( &startB[b] );
        for ( int i = 0; i < cIterations; i++ ) {
            NullKernel<<<1,blockSize[b]>>>();
            cudaDeviceSynchronize();
        }
        chTimerGetTime( &stopB[b] );
    }

    // iterate along grid and block sizes
    for (int m = 0; m < 10; m++) {
        chTimerGetTime( &startM[m] );
        for ( int i = 0; i < cIterations; i++ ) {
            NullKernel<<<gridSize[m],blockSize[m]>>>();
            cudaDeviceSynchronize();
        }
        chTimerGetTime( &stopM[m] );
    }

    {
        for (int g = 0; g < 10; g++) {
            printf( "1x%d,", gridSize[g] ); fflush( stdout );
        }
        printf( "\n" ); fflush( stdout );
        for (int g = 0; g < 10; g++) {
            double microseconds = 1e6*chTimerElapsedTime( &startG[g], &stopG[g] );
            double usPerLaunch = microseconds / (float) cIterations;

            printf( "%.2f,", usPerLaunch ); fflush( stdout );
        }
        printf( "\n\n" ); fflush( stdout );

        for (int b = 0; b < 10; b++) {
            printf( "%dx1,", blockSize[b] ); fflush( stdout );
        }
        printf( "\n" ); fflush( stdout );
        for (int b = 0; b < 10; b++) {
            double microseconds = 1e6*chTimerElapsedTime( &startB[b], &stopB[b] );
            double usPerLaunch = microseconds / (float) cIterations;
            printf( "%.2f,", usPerLaunch ); fflush( stdout );
        }
        printf( "\n\n" ); fflush( stdout );

        for (int m = 0; m < 10; m++) {
            printf( "%dx%d,", gridSize[m], blockSize[m] ); fflush( stdout );
        }
        printf( "\n" ); fflush( stdout );
        for (int m = 0; m < 10; m++) {
            double microseconds = 1e6*chTimerElapsedTime( &startM[m], &stopM[m] );
            double usPerLaunch = microseconds / (float) cIterations;
            printf( "%.2f,", usPerLaunch ); fflush( stdout ); 
        }
        printf( "\n\n" ); fflush( stdout );
    }

    return 0;
}
