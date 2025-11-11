#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

# Output file for strided memory access results
OUTPUT_FILE="strided_memory_access_results.txt"

# Stride values to test (5 values between 1 and 32)
STRIDES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20 32 50 64 90 128 151 256 301 512 1024)

# Constant parameters
THREADS_PER_BLOCK=1024
THREADBLOCKS=16

# Number of iterations for stable results
ITERATIONS=100

# Write CSV header
echo "stride,threads_per_block,threadblocks,bandwidth_gbps" > $OUTPUT_FILE

# Run parameter sweep
for stride in "${STRIDES[@]}"; do
    # Run benchmark - output is CSV format: stride,threads_per_block,threadblocks,bandwidth
    # -t 1024: constant threads per block
    # -g 1: constant number of threadblocks
    # --stride $stride: varying stride value
    ./template/bin/memCpy --global-stride --stride $stride -t $THREADS_PER_BLOCK -g $THREADBLOCKS -i $ITERATIONS -y 2>/dev/null >> $OUTPUT_FILE
done

echo "Benchmark complete. Results saved to $OUTPUT_FILE"