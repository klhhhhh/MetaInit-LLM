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


torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv_id=gpt_124m_$SLURM_JOB_ID\
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /u/klin4/MetaInit-LLM/training/megatron_gpt_pretraining.py  \
    --config-path=/u/klin4/MetaInit-LLM/conf \
    --config-name=megatron_gpt_config \
    trainer.devices=4 \
    trainer.num_nodes=8 \
    trainer.max_epochs=null \
    trainer.max_steps=50000 \
    trainer.val_check_interval=250 \
    trainer.log_every_n_steps=25 \
    trainer.limit_val_batches=50 \
    trainer.limit_test_batches=50 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=16 \
    model.transformer_engine=True \
    model.megatron_amp_O2=False \
    model.micro_batch_size=96 \
    model.global_batch_size=3072 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.max_position_embeddings=1024 \
    model.encoder_seq_length=1024 \
    model.hidden_size=768 \
    model.ffn_hidden_size=3072 \
    model.num_layers=12 \
    model.num_attention_heads=12 \
    model.init_method_std=0.021 \
    model.hidden_dropout=0.1 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.vocab_file=/work/hdd/bdrw/klin4/wiki/gpt2-vocab.json \
    model.tokenizer.merge_file=/work/hdd/bdrw/klin4/wiki/gpt2-merges.txt \
    model.data.data_prefix=[1,/work/hdd/bdrw/klin4/openwebtext/gpt2_openwebtext_text_document] \
    model.data.num_workers=2 \
    model.data.seq_length=1024 \
    model.data.splits_string=\'980,10,10\' \
    model.optim.name=fused_adam \
    model.optim.lr=6e-4 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.1 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=750 \
    model.optim.sched.constant_steps=80000 \
    model.optim.sched.min_lr=6e-5 \
    exp_manager.exp_dir=/work/hdd/bdrw/klin4/run_gpt_124m \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.project="gpt124m" \
    exp_manager.wandb_logger_kwargs.name="run_$SLURM_JOB_ID" \
    # exp_manager.wandb_logger_kwargs.id="q2jzsjkb" \
    # exp_manager.wandb_logger_kwargs.resume="allow" \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.dirpath=/work/hdd/bdrw/klin4/run_gpt_124m/megatron_gpt/checkpoints \
    exp_manager.checkpoint_callback_params.monitor=val_loss \
    exp_manager.checkpoint_callback_params.save_top_k=200 \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.checkpoint_callback_params.always_save_nemo=True
