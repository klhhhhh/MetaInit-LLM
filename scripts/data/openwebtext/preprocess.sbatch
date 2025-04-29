#!/bin/bash  
#SBATCH --account=bdrw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBTACH --mem=64g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nankaicslk@gmail.com

source /u/klin4/envs/build_nemo.sh
eval "$(conda shell.bash hook)"
conda activate nemo

python /u/klin4/MetaInit-LLM/scripts/data/openwebtext/data_clean.py

ROOT="/work/hdd/bdrw/klin4/openwebtext"  

cd /u/klin4/NeMo-modular-training
python ./scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
--input="$ROOT/openwebtext_cleaned.jsonl" \
--json-keys=text \
--tokenizer-library=megatron \
--vocab /work/hdd/bdrw/klin4/wiki/gpt2-vocab.json \
--dataset-impl mmap \
--tokenizer-type GPT2BPETokenizer \
--merge-file /work/hdd/bdrw/klin4/wiki/gpt2-merges.txt \
--output-prefix=/work/hdd/bdrw/klin4/openwebtext/gpt2_openwebtext \
--append-eod \
--workers=32