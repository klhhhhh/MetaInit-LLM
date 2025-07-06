export NVIDIA_PYTORCH_VERSION=24.12

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  /u/klin4/MetaInit-LLM/training/projection_init_pretraining.py \
