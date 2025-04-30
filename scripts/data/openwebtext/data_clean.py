import json

input_path = '/work/hdd/bdrw/klin4/openwebtext/openwebtext.jsonl'

output_path = '/work/hdd/bdrw/klin4/openwebtext/openwebtext_cleaned.jsonl'

with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for idx, line in enumerate(fin):
        text = line.strip()
        if not text:
            continue
        json_obj = {"text": text}
        fout.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
