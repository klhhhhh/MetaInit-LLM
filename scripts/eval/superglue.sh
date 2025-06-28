export WANDB_API_KEY=54c49dff7abb6ed19894a8aaec8b305d316f0072
export HF_HOME=/pscratch/sd/k/klhhhhh/hf_cache_lm_eval
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/models
export HF_DATASETS_TRUST_REMOTE_CODE=1

source /pscratch/sd/k/klhhhhh/envs/nemo/bin/activate
source /global/homes/k/klhhhhh/NeMo-modular-training/modular-training/scripts/gpt/export_package.sh

torchrun --nproc-per-node=1 --no-python lm_eval --model nemo_lm \
  --model_args path='/pscratch/sd/k/klhhhhh/checkpoints/nemo/exp/unzip_350m',data_parallel_size=1  \
  --wandb_args project=lm-eval-harness-gpt-super-glue-lm-eval-v1-350m \
  --tasks super-glue-lm-eval-v1 \
  --batch_size 32 \
  --output_path /pscratch/sd/k/klhhhhh/checkpoints/nemo/exp/super-glue-lm-eval-v1_350m \
  --log_samples