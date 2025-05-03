# mkdir -p /work/hdd/bdrw/klin4/checkpoints/nemo/unzipped_checkpoint/gpt124m
# tar -xvf /work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo -C /work/hdd/bdrw/klin4/checkpoints/nemo/unzipped_checkpoint/gpt124m

torchrun --nproc-per-node=1 /u/klin4/lm-evaluation-harness/main.py --model nemo_lm \
  --model_args path='/work/hdd/bdrw/klin4/checkpoints/nemo/unzipped_checkpoint/gpt124m',devices=1 \
  --tasks lambada_openai,super-glue-lm-eval-v1,winogrande \
  --batch_size 96