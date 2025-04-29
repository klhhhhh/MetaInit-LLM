#!/bin/bash  
#SBATCH --account=bdrw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBTACH --mem=128g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nankaicslk@gmail.com

source /u/klin4/envs/build_nemo.sh
eval "$(conda shell.bash hook)"
conda activate nemo


# Set the ROOT path  
ROOT="/work/hdd/bdrw/klin4/wiki/text"  

# Check if the wiki_all.json file exists, if so, delete it  
if [ -f "$ROOT/wiki_all.json" ]; then  
        rm "$ROOT/wiki_all.json"  
fi  
    
# Create an empty wiki_all.json file  
touch "$ROOT/wiki_all.json"
    
# Traverse all files in the ROOT path  
find $ROOT -type f -name "*" -print0 | while IFS= read -r -d $'\0' file; do  
        # Append all file contents to the wiki_all.json file  
        cat "$file" >> "$ROOT/wiki_all.json"  
done  

cd /u/klin4/NeMo-modular-training
python ./scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
--input="$ROOT/wiki_all.json" \
--json-keys=text \
--tokenizer-library=megatron \
--vocab /work/hdd/bdrw/klin4/wiki/gpt2-vocab.json \
--dataset-impl mmap \
--tokenizer-type GPT2BPETokenizer \
--merge-file /work/hdd/bdrw/klin4/wiki/gpt2-merges.txt \
--output-prefix=/work/hdd/bdrw/klin4/wiki/hfbpe_gpt_training_data \
--append-eod \
--workers=32