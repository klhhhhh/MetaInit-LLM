#!/bin/bash

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
NNODES=1
NPROC_PER_NODE=1
SCRIPT_PATH="$PROJECT_DIR/training/projection_init_pretraining.py"

# Parameters passed to Python
SMALL_MODEL_PATH="/work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo"
LARGE_MODEL_CFG_NAME="megatron_gpt_350m_config"
DEVICE="cuda"
RANK=64
LEARNABLE=""

# Enable learnable if needed
# LEARNABLE="--learnable"

# Execute
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  "$SCRIPT_PATH" \
  --small_model_path "$SMALL_MODEL_PATH" \
  --large_model_cfg_name "$LARGE_MODEL_CFG_NAME" \
  --device "$DEVICE" \
  --rank "$RANK" \
  $LEARNABLE
