#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

# Output file for streamed n-body results
OUTPUT_FILE="nbody_limited_nonstreamed_kernel_results.txt"

# Fixed number of particles
NUM_ELEMENTS=512000  # 512k particles

# Number of iterations (can be adjusted)
NUM_ITERATIONS=10

# Threads per block (can be adjusted, default is 128)
THREADS_PER_BLOCK=128

# Segment sizes to test (in bytes, up to 4/3 * 1MB = 1,398,101 bytes)
# Testing various sizes from 256KB up to the limit
SEGMENT_SIZES=(262144 393216 524288 655360 786432 917504 1048576 1179648 1310720 1398101)
# 256KB, 384KB, 512KB, 640KB, 768KB, 896KB, 1MB, 1.125MB, 1.25MB, ~1.33MB (4/3 MB)

# Write CSV header
echo "num_elements,segment_size_bytes,segment_size_kb,num_iterations,threads_per_block,h2d_time_ms,h2d_bandwidth_gbs,kernel_time_ms,d2h_time_ms,d2h_bandwidth_gbs,total_time_ms" > $OUTPUT_FILE

# Run parameter sweep
for segment_size in "${SEGMENT_SIZES[@]}"; do
    # Run streamed nbody with --silent flag to suppress iteration output
    OUTPUT=$(./bin/nbody_streamed --silent -p -s $NUM_ELEMENTS -i $NUM_ITERATIONS -t $THREADS_PER_BLOCK --segment-size $segment_size 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    
    # Parse using awk - extract values from lines matching patterns
    num_elements_parsed=$(echo "$OUTPUT" | awk '/Num Elements:/ {print $4}')
    num_iterations_parsed=$(echo "$OUTPUT" | awk '/Num Iterations:/ {print $4}')
    threads_per_block_parsed=$(echo "$OUTPUT" | awk '/Threads per block:/ {print $5}')
    h2d_time=$(echo "$OUTPUT" | awk '/Time to Copy to Device:/ {print $6}')
    # Bandwidth appears on the line immediately after "Time to Copy to Device"
    h2d_bandwidth=$(echo "$OUTPUT" | awk '/Time to Copy to Device:/ {getline; if (/Copy Bandwidth:/) print $4}')
    kernel_time=$(echo "$OUTPUT" | awk '/Time for n-Body Computation:/ {print $6}')
    d2h_time=$(echo "$OUTPUT" | awk '/Time to Copy from Device:/ {print $6}')
    # Bandwidth appears on the line immediately after "Time to Copy from Device"
    d2h_bandwidth=$(echo "$OUTPUT" | awk '/Time to Copy from Device:/ {getline; if (/Copy Bandwidth:/) print $4}')
    total_time=$(echo "$OUTPUT" | awk '/Time to Finnish:/ {print $5}')
    
    # Calculate segment size in KB
    segment_size_kb=$((segment_size / 1024))
    
    # Write to output file if we got the required values
    if [ -n "$num_elements_parsed" ] && [ -n "$total_time" ]; then
        num_iterations_parsed=${num_iterations_parsed:-$NUM_ITERATIONS}
        threads_per_block_parsed=${threads_per_block_parsed:-$THREADS_PER_BLOCK}
        h2d_time=${h2d_time:-"N/A"}
        h2d_bandwidth=${h2d_bandwidth:-"N/A"}
        kernel_time=${kernel_time:-"N/A"}
        d2h_time=${d2h_time:-"N/A"}
        d2h_bandwidth=${d2h_bandwidth:-"N/A"}
        echo "$num_elements_parsed,$segment_size,$segment_size_kb,$num_iterations_parsed,$threads_per_block_parsed,$h2d_time,$h2d_bandwidth,$kernel_time,$d2h_time,$d2h_bandwidth,$total_time" >> $OUTPUT_FILE
    else
        echo "Warning: Failed to parse output for segment_size=$segment_size" >&2
        echo "Output was:" >&2
        echo "$OUTPUT" >&2
    fi
done

echo "Benchmark complete. Results saved to $OUTPUT_FILE"

