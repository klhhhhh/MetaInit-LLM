torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    /u/klin4/MetaInit-LLM/nemo_utils/model_loader.py \
    --nemo_path /work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo