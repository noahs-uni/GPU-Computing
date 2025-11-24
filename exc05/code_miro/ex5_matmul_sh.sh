#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

# Output file for matrix multiplication results
OUTPUT_FILE="threads_per_block_results_L1.txt"
#OUTPUT_FILE="matmul_shared_results.txt"
# Matrix sizes to test (10 values)
MATRIX_SIZES=(1024 1536 2048 2560 3072 4096 8192)

# Threads per block to test (10 values, must be <= 32)
# Default is 32 if not specified (see DEFAULT_BLOCK_DIM in main.cu)
THREADS_PER_BLOCK=(32)

# Number of runs to average over for each (size, threads) pair
RUNS=5

# Write CSV header
echo "matrix_size,threads_per_block,H2D_time_ms,kernel_time_ms,D2H_time_ms,CPU_time_ms,results_match" > $OUTPUT_FILE

# Run parameter sweep
# Note: Not using -c flag so checking is enabled and CPU is timed
for size in "${MATRIX_SIZES[@]}"; do
    for threads in "${THREADS_PER_BLOCK[@]}"; do
        sum_h2d=0
        sum_d2h=0
        sum_kernel=0
        sum_cpu=0
        cpu_samples=0
        successful_runs=0
        matrix_size=""
        threads_per_block=""
        results_match_overall="YES"

        for ((run=1; run<=RUNS; run++)); do
            # Run matMul and parse output
            OUTPUT=$(./bin/matMul -c -s $size -t $threads 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
            
            # Parse using awk - extract values from lines matching patterns
            parsed_matrix_size=$(echo "$OUTPUT" | awk '/Matrix Size:/ {print $4}')
            parsed_threads=$(echo "$OUTPUT" | awk '/Block Dim:/ {print $4}' | awk -F'x' '{print $1}')
            h2d_time=$(echo "$OUTPUT" | awk '/Time to Copy to Device:/ {print $7}')
            d2h_time=$(echo "$OUTPUT" | awk '/Time to Copy from Device:/ {print $7}')
            kernel_time=$(echo "$OUTPUT" | awk '/Time for Matrix Multiplication:/ && !/CPU/ {print $6}')
            cpu_time=$(echo "$OUTPUT" | awk '/Time for CPU Matrix Multiplication:/ {print $7}')
            results_match=$(echo "$OUTPUT" | awk '/Results Match:/ {print $4}')

            if [ -z "$parsed_matrix_size" ] || [ -z "$parsed_threads" ] || [ -z "$h2d_time" ] || [ -z "$kernel_time" ] || [ -z "$d2h_time" ]; then
                echo "Warning: Failed to parse output for size=$size threads=$threads (run $run)" >&2
                echo "$OUTPUT" >&2
                continue
            fi

            matrix_size=${matrix_size:-$parsed_matrix_size}
            threads_per_block=${threads_per_block:-$parsed_threads}

            sum_h2d=$(echo "$sum_h2d + $h2d_time" | bc -l)
            sum_d2h=$(echo "$sum_d2h + $d2h_time" | bc -l)
            sum_kernel=$(echo "$sum_kernel + $kernel_time" | bc -l)
            successful_runs=$((successful_runs + 1))

            if [ -n "$cpu_time" ]; then
                sum_cpu=$(echo "$sum_cpu + $cpu_time" | bc -l)
                cpu_samples=$((cpu_samples + 1))
            fi

            if [ -n "$results_match" ]; then
                if [ "$results_match" != "YES" ]; then
                    results_match_overall="NO"
                fi
            else
                results_match_overall="N/A"
            fi
        done

        if [ -n "$matrix_size" ] && [ -n "$threads_per_block" ] && [ $successful_runs -gt 0 ]; then
            avg_h2d=$(echo "scale=6; $sum_h2d / $successful_runs" | bc -l)
            avg_d2h=$(echo "scale=6; $sum_d2h / $successful_runs" | bc -l)
            avg_kernel=$(echo "scale=6; $sum_kernel / $successful_runs" | bc -l)
            if [ $cpu_samples -gt 0 ]; then
                avg_cpu=$(echo "scale=6; $sum_cpu / $cpu_samples" | bc -l)
            else
                avg_cpu="N/A"
            fi

            echo "$matrix_size,$threads_per_block,$avg_h2d,$avg_kernel,$avg_d2h,$avg_cpu,$results_match_overall" >> $OUTPUT_FILE
        else
            echo "Skipping entry for size=$size threads=$threads due to missing data" >&2
        fi
    done
done

echo "Benchmark complete. Results saved to $OUTPUT_FILE"

