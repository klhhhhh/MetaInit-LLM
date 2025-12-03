python /u/klin4/MetaInit-LLM/nemo_utils/read_optimizer.py \
  --cfg /u/klin4/MetaInit-LLM/conf/megatron_gpt_124m_weight_mapping_config.yaml \
  --ckpt-dir /work/hdd/bdrw/klin4/run_gpt_124m_test/megatron_gpt/checkpoints/megatron_gpt--val_loss=3.15-step=50000-consumed_samples=153600000.0-last \
  --save-optimizer /work/hdd/bdrw/klin4/optimizer_ckpt/optimizer_states_step10000.pt