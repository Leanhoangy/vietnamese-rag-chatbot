"""
Tạo test triplets từ test_set.json dùng random negatives.
Lưu dưới dạng HuggingFace dataset để trainer load được.
"""
import json
import random
from datasets import Dataset

random.seed(42)

with open("test_set.json", encoding="utf-8") as f:
    test_cases = json.load(f)

triplets = []
for i, case in enumerate(test_cases):
    query    = case["question"]
    positive = case.get("ground_truth", "")
    if isinstance(positive, list):
        positive = positive[0] if positive else ""
    if not positive or len(positive) < 10:
        continue

    # 3 random negatives từ các câu khác
    others = [c for j, c in enumerate(test_cases) if j != i]
    sampled = random.sample(others, min(3, len(others)))
    for neg_case in sampled:
        neg = neg_case.get("ground_truth", "")
        if isinstance(neg, list):
            neg = neg[0] if neg else ""
        if neg and len(neg) > 10 and neg != positive:
            triplets.append({"anchor": query, "positive": positive, "negative": neg})

ds = Dataset.from_list(triplets)
ds.save_to_disk("test_triplets_dataset")
print(f"Created {len(triplets)} test triplets → test_triplets_dataset/")
