#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

OUTPUT_FILE="7_1.csv"

./bin/nbody -s 2 -t 1 -i 100 --o --v