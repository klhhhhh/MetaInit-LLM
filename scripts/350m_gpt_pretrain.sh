#!/bin/bash

source /pscratch/sd/k/klhhhhh/envs/nemo/bin/activate
bash /global/homes/k/klhhhhh/NeMo-modular-training/modular-training/scripts/gpt/export_package.sh

torchrun \
    --nnodes=16 \
    --nproc_per_node=4 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv_id=gpt_350m \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /global/homes/k/klhhhhh/NeMo-modular-training/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path=/global/homes/k/klhhhhh/NeMo-modular-training/examples/nlp/language_modeling/conf \
    --config-name=megatron_gpt_config \
    trainer.devices=4 \
    trainer.num_nodes=16 \
    trainer.max_epochs=null \
    trainer.max_steps=300000 \
    trainer.val_check_interval=3000 \
    trainer.log_every_n_steps=25 \
    trainer.limit_val_batches=50 \
    trainer.limit_test_batches=50 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=16 \
    model.transformer_engine=True \
    model.megatron_amp_O2=False \
    model.micro_batch_size=64 \
    model.global_batch_size=256 \
    model.tensor_model_parallel_size=4 \
    model.pipeline_model_parallel_size=4 \
    model.max_position_embeddings=1024 \
    model.encoder_seq_length=1024 \
    model.hidden_size=1024 \
    model.ffn_hidden_size=4096 \
    model.num_layers=24 \
    model.num_attention_heads=16 \
    model.init_method_std=0.021 \
    model.hidden_dropout=0.1 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.vocab_file=gpt2-vocab.json \
    model.tokenizer.merge_file=gpt2-merges.txt \
    model.data.data_prefix=[0.5,/pscratch/sd/k/klhhhhh/dataset/nemo/wiki/hfbpe_gpt_training_data_text_document,0.5,/pscratch/sd/k/klhhhhh/openwebtext_data/my-gpt2-oepnwebtext_text_document] \
    model.data.num_workers=2 \
    model.data.seq_length=1024 \
    model.data.splits_string=\'980,10,10\' \
    model.optim.name=fused_adam \
    model.optim.lr=3e-4 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.1 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=2000 \
    model.optim.sched.constant_steps=80000 \
    model.optim.sched.min_lr=6e-5 \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.dirpath=/pscratch/sd/k/klhhhhh/checkpoints/nemo/gpt_350m \
    exp_manager.checkpoint_callback_params.monitor=val_loss \
    exp_manager.checkpoint_callback_params.save_top_k=5 \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.checkpoint_callback_params.always_save_nemo=True