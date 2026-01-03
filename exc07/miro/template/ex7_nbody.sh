#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

# Output file for n-body results
OUTPUT_FILE="nbody_benchmark_results.txt"

# Object counts to test (numElements)
OBJECT_COUNTS=(64 128 256 512 1024 2048 4096 8192 16384 32768)

# Number of iterations (can be adjusted)
NUM_ITERATIONS=1000

# Threads per block (can be adjusted, default is 128)
THREADS_PER_BLOCK=128

# Write CSV header
echo "num_elements,num_iterations,threads_per_block,H2D_time_ms,H2D_bandwidth_GBs,D2H_time_ms,D2H_bandwidth_GBs,kernel_time_ms" > $OUTPUT_FILE

# Run parameter sweep
for num_elements in "${OBJECT_COUNTS[@]}"; do
    # Run nbody with --silent flag to suppress iteration output
    OUTPUT=$(./bin/nbody --silent -p -s $num_elements -i $NUM_ITERATIONS -t $THREADS_PER_BLOCK 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    
    # Parse using awk - extract values from lines matching patterns
    num_elements_parsed=$(echo "$OUTPUT" | awk '/Num Elements:/ {print $4}')
    num_iterations_parsed=$(echo "$OUTPUT" | awk '/Num Iterations:/ {print $4}')
    threads_per_block_parsed=$(echo "$OUTPUT" | awk '/Threads per block:/ {print $5}')
    h2d_time=$(echo "$OUTPUT" | awk '/Time to Copy to Device:/ {print $7}')
    # Bandwidth appears on the line immediately after "Time to Copy to Device"
    h2d_bandwidth=$(echo "$OUTPUT" | awk '/Time to Copy to Device:/ {getline; if (/Copy Bandwidth:/) print $4}')
    d2h_time=$(echo "$OUTPUT" | awk '/Time to Copy from Device:/ {print $7}')
    # Bandwidth appears on the line immediately after "Time to Copy from Device"
    d2h_bandwidth=$(echo "$OUTPUT" | awk '/Time to Copy from Device:/ {getline; if (/Copy Bandwidth:/) print $4}')
    kernel_time=$(echo "$OUTPUT" | awk '/Time for n-Body Computation:/ {print $6}')
    
    # Write to output file if we got the required values
    if [ -n "$num_elements_parsed" ] && [ -n "$h2d_time" ] && [ -n "$kernel_time" ] && [ -n "$d2h_time" ]; then
        num_iterations_parsed=${num_iterations_parsed:-$NUM_ITERATIONS}
        threads_per_block_parsed=${threads_per_block_parsed:-$THREADS_PER_BLOCK}
        h2d_bandwidth=${h2d_bandwidth:-"N/A"}
        d2h_bandwidth=${d2h_bandwidth:-"N/A"}
        echo "$num_elements_parsed,$num_iterations_parsed,$threads_per_block_parsed,$h2d_time,$h2d_bandwidth,$d2h_time,$d2h_bandwidth,$kernel_time" >> $OUTPUT_FILE
    else
        echo "Warning: Failed to parse output for num_elements=$num_elements" >&2
        echo "Output was:" >&2
        echo "$OUTPUT" >&2
    fi
done

echo "Benchmark complete. Results saved to $OUTPUT_FILE"

