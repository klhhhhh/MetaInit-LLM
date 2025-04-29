from datasets import load_dataset

dataset = load_dataset('openwebtext')

with open('openwebtext.jsonl', 'w', encoding='utf-8') as f:
    for example in dataset['train']:
        text = example['text'].replace('\n', ' ')
        f.write(text.strip() + '\n')