#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

OUTPUT_FILE="5_3.csv"

rm $OUTPUT_FILE
touch $OUTPUT_FILE
cat header.csv >> $OUTPUT_FILE

# Threads-per-block (t x t <= 1k)
T_LIST=(32)

# Problem sizes
S_LIST=(800 1024 4096 5000 8000 8192)

ITERATIONS=10

for T in "${T_LIST[@]}"; do
    for S in "${S_LIST[@]}"; do
        echo "Running with T=$T, S=$S"
        for ((i=0; i<$ITERATIONS; i++)); do
          ./bin/matMul -s $S -t $T --shared --no-checking >> $OUTPUT_FILE
        done
    done
done
