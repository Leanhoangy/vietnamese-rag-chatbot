"""Tạo test_set_clean.json: các câu hỏi trong test_set.json KHÔNG xuất hiện trong tập triplet huấn luyện.

train_custom.py từng tạo triplet từ test_set.json, nên phần lớn câu test đã lọt vào legal_triplets_dataset
(dataset dùng để fine-tune E5). Đánh giá trên phần còn lại để số liệu không bị thổi phồng.

Chạy từ thư mục gốc project (cần legal_triplets_dataset/ ở local):
    python evaluation/make_clean_test_set.py
"""

import json
import re
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent.parent


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


test_set = json.loads((ROOT / "test_set.json").read_text(encoding="utf-8"))
training_anchors = {_norm(a) for a in load_from_disk(str(ROOT / "legal_triplets_dataset"))["anchor"]}

clean = [case for case in test_set if _norm(case["question"]) not in training_anchors]
(ROOT / "test_set_clean.json").write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(clean)}/{len(test_set)} test questions never appear in the training triplets → test_set_clean.json")
