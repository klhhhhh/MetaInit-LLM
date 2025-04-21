#!/bin/bash  
    
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

cd /u/klin4/Megatron-modular-training
python tools/preprocess_data.py \
--input "$ROOT/wiki_all.json" \
--output-prefix my-gpt2 \
--dataset-impl mmap \
--tokenizer-type GPT2BPETokenizer   \
--append-eod  \
--vocab-file /work/hdd/bdrw/klin4/wiki/gpt2-vocab.json \
--merge-file /work/hdd/bdrw/klin4/wiki/gpt2-merges.txt  \
--workers 16 \
--partitions 16