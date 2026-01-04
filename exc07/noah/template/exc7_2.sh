#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

OUTPUT_FILE="7_1.csv"

echo "size,blockSize,iterations,time,ips" > $OUTPUT_FILE
# Problem sizes
S_LIST=(15000 18000 21000 24000 27000 30000)
#T_LIST=(1 4 16 64 128 256 512 1024)
T_LIST=(16 64 128 256 512)

# Execute 10 times for each size in S_LIST
for threads in "${T_LIST[@]}"; do
  for size in "${S_LIST[@]}"; do
    echo "./bin/nbody -s ${size} -t ${threads} -i 500 --silent --o"
    for i in {1..4}; do
      ./bin/nbody -s ${size} -t ${threads} -i 500 --silent --o >> $OUTPUT_FILE
    done
  done
done