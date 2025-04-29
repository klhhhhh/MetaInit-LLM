#!/bin/bash
#SBATCH --account=bdrw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nankaicslk@gmail.com

# allocate MASTER_PORT
if [ "$SLURM_PROCID" -eq 0 ]; then
  find_free_port() {
    while true; do
      PORT=$(shuf -i 49152-65535 -n 1)
      ss -tuln | grep ":$PORT " > /dev/null
      if [ $? -ne 0 ]; then
        echo $PORT
        return
      fi
    done
  }
  export MASTER_PORT=$(find_free_port)
  echo "Master Port: $MASTER_PORT"
fi

# sync MASTER_PORT
MASTER_PORT=$(srun --ntasks=1 echo $MASTER_PORT)
export MASTER_PORT

# set MASTER_ADDR IP 
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

export NVIDIA_PYTORCH_VERSION=24.12

srun --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,NVIDIA_PYTORCH_VERSION=$NVIDIA_PYTORCH_VERSION \
     bash /u/klin4/MetaInit-LLM/scripts/test.sh