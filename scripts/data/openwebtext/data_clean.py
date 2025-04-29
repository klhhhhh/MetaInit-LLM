import json

input_path = '/work/hdd/bdrw/klin4/openwebtext/openwebtext.jsonl'
output_path = '/work/hdd/bdrw/klin4/openwebtext/clean_openwebtext.jsonl'

with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for idx, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            fout.write(json.dumps(obj) + '\n')
        except json.JSONDecodeError:
            print(f"Warning: Skipping non-JSON line at {idx}")
            continue
