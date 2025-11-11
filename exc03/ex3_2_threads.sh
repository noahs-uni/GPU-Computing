#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

# Output file for coalesced memory access results
OUTPUT_FILE="coalesced_memory_access_results.txt"

# Data sizes to test (in bytes) - from 1KB to 1GB
DATA_SIZES=(1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 268435456 1073741824)

# Threads per block to test
THREADS_PER_BLOCK=(32 64 128 256 512 1024)

# Number of iterations for stable results
ITERATIONS=100

# Write CSV header
echo "size_bytes,threads_per_block,bandwidth_gbps" > $OUTPUT_FILE

# Run parameter sweep
for size in "${DATA_SIZES[@]}"; do
    for threads in "${THREADS_PER_BLOCK[@]}"; do
        # Run benchmark - output is CSV format: size,threads,bandwidth
        ./template/bin/memCpy --global-coalesced -s $size -t $threads -g 1 -i $ITERATIONS -y 2>/dev/null >> $OUTPUT_FILE
    done
done

echo "Benchmark complete. Results saved to $OUTPUT_FILE"