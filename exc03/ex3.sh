#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -p exercise-gpu
#SBATCH -o ex3_out_D2H.txt

bin/Data_trans_D2H_miro
