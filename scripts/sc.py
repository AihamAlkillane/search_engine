import ir_datasets
import os
import json

dataset_name = "antique/test"  # جرب dataset مختلفة

dataset = ir_datasets.load(dataset_name)

os.makedirs(f"data/{dataset_name.replace('/', '_')}", exist_ok=True)

with open(f"data/{dataset_name.replace('/', '_')}/docs.jsonl", "w", encoding="utf-8") as f:
    for i, doc in enumerate(dataset.docs_iter()):
        json_line = json.dumps(doc._asdict(), ensure_ascii=False)
        f.write(json_line + "\n")
        if i >= 1000:  # جرب تكتب أول 1000 وثيقة فقط عشان تختبر
            break
