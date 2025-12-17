#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

OUTPUT_FILE="6_3.csv"

# Header: s = problem size, t = threads-per-block, then the program output fields,
# plus total_ms = memcpy_h2d_ms + memcpy_d2h_ms + t_matmul_ms
echo "array_size,block_size,time,bandwidth" > $OUTPUT_FILE

# Problem sizes
S_LIST=(16 64 256 1024 4096 16384 65536 262144 1048576 2097152)
B_LIST=(1 8 16 64 128 512 1024)

# Execute 10 times for each size in S_LIST
for size in "${S_LIST[@]}"; do
  for block in "${B_LIST[@]}"; do
    echo "./bin/reduction -s ${size} -t ${block}"
    for i in {1..10}; do
      ./bin/reduction -s "$size" -t "$block" >> $OUTPUT_FILE
    done
  done
done