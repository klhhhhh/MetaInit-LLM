import json

# 输入你的坏文本文件
input_path = '/work/hdd/bdrw/klin4/openwebtext/openwebtext.jsonl'

# 输出干净的合法jsonl文件
output_path = '/work/hdd/bdrw/klin4/openwebtext/openwebtext_cleaned.jsonl'

with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for idx, line in enumerate(fin):
        text = line.strip()
        if not text:
            continue  # 跳过空行
        json_obj = {"text": text}
        fout.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
