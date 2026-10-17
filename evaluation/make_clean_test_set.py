"""Tạo test_set_clean.json: các câu hỏi trong test_set.json KHÔNG xuất hiện trong tập triplet huấn luyện
và không lấy đáp án từ Nghị định 100/2019.

train_custom.py từng tạo triplet từ test_set.json, nên phần lớn câu test đã lọt vào legal_triplets_dataset
(dataset dùng để fine-tune E5). Đánh giá trên phần còn lại để số liệu không bị thổi phồng.
Câu theo Nghị định 100/2019 bị loại vì mức phạt đường bộ đã được thay bằng Nghị định 168/2024 — chatbot
trả lời đúng theo văn bản mới sẽ bị chấm lệch đáp án cũ. Mảng giao thông được kiểm tra bằng
evaluation/golden_set.json (đáp án theo Nghị định 168/2024, 238/2026).

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


test_set = json.loads((ROOT / "data/qa/test_set.json").read_text(encoding="utf-8"))
training_anchors = {_norm(a) for a in load_from_disk(str(ROOT / "data/triplets/legal_triplets_dataset"))["anchor"]}

_DECREE_100 = re.compile(r"Nghidinh 100|Nghị định 100/2019")

unseen = [case for case in test_set if _norm(case["question"]) not in training_anchors]
clean = [case for case in unseen if not _DECREE_100.search(case.get("source", ""))]
(ROOT / "data/qa/test_set_clean.json").write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(unseen)}/{len(test_set)} test questions never appear in the training triplets; "
      f"{len(unseen) - len(clean)} answered from Decree 100/2019 dropped → {len(clean)} in test_set_clean.json")
