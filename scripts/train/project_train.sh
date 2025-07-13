#!/bin/bash

eval "$(conda shell.bash hook)"
source /u/klin4/envs/build_nemo.sh
conda activate nemo

export PATH="/u/klin4/.conda/envs/nemo/bin:$PATH"
export WANDB_API_KEY=54c49dff7abb6ed19894a8aaec8b305d316f0072

RDZV_FILE="$HOME/.rdzv/rdzv_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname $RDZV_FILE)"

# rank 0
if [ "$SLURM_PROCID" -eq 0 ]; then
  MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1 )
  MASTER_PORT=$(shuf -i 49152-65535 -n 1)
  echo "$MASTER_ADDR $MASTER_PORT" > "$RDZV_FILE"
fi

# all nodes wait
while [ ! -f "$RDZV_FILE" ]; do sleep 1; done

# all nodes read
read MASTER_ADDR MASTER_PORT < "$RDZV_FILE"
export MASTER_ADDR
export MASTER_PORT

echo "$MASTER_ADDR"
echo "$MASTER_PORT"

export NVIDIA_PYTORCH_VERSION=24.12

# Project root directory
PROJECT_DIR="/u/klin4/MetaInit-LLM"

# Switch to the project root directory
cd "$PROJECT_DIR" || {
  echo "❌ Failed to cd into $PROJECT_DIR"
  exit 1
}

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
echo "✅ PYTHONPATH set to: $PYTHONPATH"

# Set default parameters
NNODES=8
NPROC_PER_NODE=4
SCRIPT_PATH="$PROJECT_DIR/training/projection_init_pretraining.py"

# Parameters passed to Python
SMALL_MODEL_PATH="/work/hdd/bdrw/klin4/run_gpt_124m/megatron_gpt/checkpoints_124m/gpt-124-last"
LARGE_MODEL_CFG_NAME="megatron_gpt_350m_config"
PROJECT_DEVICE="cpu"
TRAIN_DEVICE="cuda"
RANK=64
LEARNABLE=""

# Enable learnable if needed
# LEARNABLE="--learnable"

# Execute
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --rdzv_id=gpt_350m_project \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  "$SCRIPT_PATH" \
  --small_model_path "$SMALL_MODEL_PATH" \
  --large_model_cfg_name "$LARGE_MODEL_CFG_NAME" \
  --project_device "$PROJECT_DEVICE" \
  --train_device "$TRAIN_DEVICE" \
  --rank "$RANK" \
  $LEARNABLE
