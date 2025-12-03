torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    /u/klin4/MetaInit-LLM/nemo_utils/model_loader.py \
    --nemo_path /work/hdd/bdrw/klin4/run_gpt350m_compare/megatron_gpt/checkpoints/megatron_gpt--val_loss=3.01-step=15000-consumed_samples=7680000.0-last \
    --config_path /u/klin4/MetaInit-LLM/conf/megatron_gpt_350m_config.yaml