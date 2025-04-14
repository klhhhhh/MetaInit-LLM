torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    /global/homes/k/klhhhhh/MetaInit-LLM/nemo_utils/model_loader.py \
    --nemo_path /pscratch/sd/k/klhhhhh/checkpoints/nemo/gpt/megatron_gpt.nemo