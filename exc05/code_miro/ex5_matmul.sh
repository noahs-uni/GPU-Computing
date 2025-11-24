#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

# Output file for matrix multiplication results
OUTPUT_FILE="test_threads_per_block_results.txt"

# Matrix sizes to test (10 values)
MATRIX_SIZES=(64 128 256 512 1024 1536 2048 2560 3072 4096)

# Threads per block to test (10 values, must be <= 32)
# Default is 32 if not specified (see DEFAULT_BLOCK_DIM in main.cu)
THREADS_PER_BLOCK=(2 4 6 8 12 16 20 24 28 32)

# Write CSV header
echo "matrix_size,threads_per_block,H2D_time_ms,kernel_time_ms,D2H_time_ms,CPU_time_ms,results_match" > $OUTPUT_FILE

# Run parameter sweep
# Note: Not using -c flag so checking is enabled and CPU is timed
for size in "${MATRIX_SIZES[@]}"; do
    for threads in "${THREADS_PER_BLOCK[@]}"; do
        # Run matMul and parse output
        OUTPUT=$(./bin/matMul -c -s $size -t $threads 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
        
        # Parse using awk - extract values from lines matching patterns
        matrix_size=$(echo "$OUTPUT" | awk '/Matrix Size:/ {print $4}')
        threads_per_block=$(echo "$OUTPUT" | awk '/Block Dim:/ {print $4}' | awk -F'x' '{print $1}')
        h2d_time=$(echo "$OUTPUT" | awk '/Time to Copy to Device:/ {print $7}')
        d2h_time=$(echo "$OUTPUT" | awk '/Time to Copy from Device:/ {print $7}')
        kernel_time=$(echo "$OUTPUT" | awk '/Time for Matrix Multiplication:/ && !/CPU/ {print $6}')
        cpu_time=$(echo "$OUTPUT" | awk '/Time for CPU Matrix Multiplication:/ {print $7}')
        results_match=$(echo "$OUTPUT" | awk '/Results Match:/ {print $4}')
        
        # Write to output file if we got the required values
        if [ -n "$matrix_size" ] && [ -n "$threads_per_block" ] && [ -n "$h2d_time" ] && [ -n "$kernel_time" ] && [ -n "$d2h_time" ]; then
            cpu_time=${cpu_time:-"N/A"}
            results_match=${results_match:-"N/A"}
            echo "$matrix_size,$threads_per_block,$h2d_time,$kernel_time,$d2h_time,$cpu_time,$results_match" >> $OUTPUT_FILE
        fi
    done
done

echo "Benchmark complete. Results saved to $OUTPUT_FILE"

