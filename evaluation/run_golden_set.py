"""Chạy bộ câu hỏi đáp án vàng: câu trả lời có đủ mức phạt, trích đúng Điều/văn bản và không dùng mức phạt cũ không.

Đáp án lấy từ Nghị định 168/2024 và 238/2026 (bản Word đã đối chiếu với bản scan chính thức).
LLM theo LLM_PROVIDER (Gemini: ~2 request/câu). Chạy từ thư mục gốc project:

    python evaluation/run_golden_set.py
"""

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["USE_FINETUNED"] = "true"

from evaluation.golden_check import check_answer  # noqa: E402
from src.llm_chain import LLM_MODEL, LLM_PROVIDER, is_transient_error  # noqa: E402
from src.retrieval_hybrid import qa_chain  # noqa: E402

GOLDEN_PATH = Path("evaluation/golden_set.json")
OUTPUT_PATH = Path("evaluation/results_golden.json")
RETRIES, WAIT_SECONDS = 3, 30
# Free tier Gemini Flash-Lite: 15 request/phút; mỗi câu ~2 request → giãn cách ~8 giây/câu
REQUEST_INTERVAL = float(os.getenv("EVAL_REQUEST_INTERVAL", "8" if LLM_PROVIDER == "gemini" else "0"))


def ask(question: str) -> dict:
    for attempt in range(RETRIES + 1):
        try:
            return qa_chain.invoke({"query": question, "chat_history": ""})
        except Exception as e:
            if is_transient_error(e) and attempt < RETRIES:
                print(f"  [Rate limit] waiting {WAIT_SECONDS}s...")
                time.sleep(WAIT_SECONDS)
            else:
                raise


items = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
results = []
for item in items:
    reply = ask(item["question"])
    problems = check_answer(reply["result"], item)
    results.append({
        "id": item["id"],
        "domain": item["domain"],
        "question": item["question"],
        "reference": item["reference"],
        "passed": not problems,
        "problems": problems,
        "answer": reply["result"],
    })
    print(f"{'PASS' if not problems else 'FAIL'}  {item['id']:28s} {'; '.join(problems)}")
    time.sleep(REQUEST_INTERVAL)

passed = sum(r["passed"] for r in results)
by_domain = {}
for r in results:
    stats = by_domain.setdefault(r["domain"], {"passed": 0, "total": 0})
    stats["passed"] += r["passed"]
    stats["total"] += 1
print(f"\n{passed}/{len(results)} golden questions passed")
for domain, stats in by_domain.items():
    print(f"  {domain:12s} {stats['passed']}/{stats['total']}")

OUTPUT_PATH.write_text(json.dumps({
    "config": {"date": date.today().isoformat(), "answer_llm": f"{LLM_PROVIDER}/{LLM_MODEL}"},
    "passed": passed,
    "total": len(results),
    "by_domain": by_domain,
    "results": results,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Results saved to: {OUTPUT_PATH}")
