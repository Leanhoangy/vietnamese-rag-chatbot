"""
Split data/qa/all_questions.json thành train_set.json và test_set.json (cùng thư mục).
Chạy từ thư mục gốc repo: python training/split_dataset.py
"""

import json
import random

TRAIN_RATIO = 0.8
SEED = 42
SOURCE = "data/qa/all_questions.json"

with open(SOURCE, encoding="utf-8") as f:
    all_data = json.load(f)

random.seed(SEED)
random.shuffle(all_data)

n_train = int(len(all_data) * TRAIN_RATIO)
train = all_data[:n_train]
test = all_data[n_train:]

with open("data/qa/train_set.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open("data/qa/test_set.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print(f"Tổng: {len(all_data)} câu")
print(f"✓ train_set.json: {len(train)} câu ({TRAIN_RATIO*100:.0f}%)")
print(f"✓ test_set.json:  {len(test)} câu ({(1-TRAIN_RATIO)*100:.0f}%)")
