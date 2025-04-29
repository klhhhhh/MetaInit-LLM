from datasets import load_dataset
import json

dataset = load_dataset('openwebtext')

with open('openwebtext.jsonl', 'w', encoding='utf-8') as f:
    for example in dataset['train']:
        text = example['text'].replace('\n', ' ').strip()
        if text:
            json_obj = {"text": text}
            f.write(json.dumps(json_obj) + '\n')