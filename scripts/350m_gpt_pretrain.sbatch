#!/bin/bash
#SBATCH --account=bdrw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=48:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nankaicslk@gmail.com

srun --export=ALL --cpu-bind=none \
     bash /u/klin4/MetaInit-LLM/scripts/350m_gpt_pretrain.sh