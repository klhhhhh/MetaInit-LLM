#!/bin/bash
#SBATCH -A m4410_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -c 8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nankaicslk@gmail.com

srun --export=ALL\
     bash /global/homes/k/klhhhhh/MetaInit-LLM/scripts/eval/superglue.sh

