# mkdir -p /work/hdd/bdrw/klin4/checkpoints/nemo/unzipped_checkpoint/gpt124m
# tar -xvf /work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo -C /work/hdd/bdrw/klin4/checkpoints/nemo/unzipped_checkpoint/gpt124m

export WANDB_API_KEY=54c49dff7abb6ed19894a8aaec8b305d316f0072
export HF_HOME=/pscratch/sd/k/klhhhhh/hf_cache_lm_eval
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/models

torchrun --nproc-per-node=1 --no-python lm_eval --model nemo_lm \
  --model_args path='/pscratch/sd/k/klhhhhh/checkpoints/nemo/exp/unzip',data_parallel_size=1  \
  --wandb_args project=lm-eval-harness-gpt-winogrande \
  --tasks lambada_openai \
  --batch_size 32 \
  --output_path /pscratch/sd/k/klhhhhh/checkpoints/nemo/exp/winogrande \
  --log_samples