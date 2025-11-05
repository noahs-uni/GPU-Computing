#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -p exercise-gpu
#SBATCH -o ex2_out.txt

/bin/nullKernelAsync
/bin/nullKernelSync
/bin/Break-even_miro
/bin/Data_trans_D2H_miro
/bin/Data_trans_H2D_miro