"""Evaluate hệ thống dùng fine-tuned e5 (bỏ qua custom transformer).

Mặc định chạy trên test_set_clean.json (câu hỏi chưa từng xuất hiện khi train). Ví dụ:

    # Gemini sinh câu trả lời (giống demo), qwen3:8b local chấm RAGAS để tiết kiệm quota
    EVAL_JUDGE_PROVIDER=ollama python evaluation/evaluate_finetuned.py --limit 3     # chạy thử
    EVAL_JUDGE_PROVIDER=ollama caffeinate -i python evaluation/evaluate_finetuned.py  # chạy hết
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--test-set", default="test_set_clean.json")
parser.add_argument("--limit", type=int, default=None, help="số câu hỏi đầu tiên cần chạy (mặc định: tất cả)")
parser.add_argument("--ragas-limit", type=int, default=None, help="số câu chấm RAGAS (mặc định: tất cả câu đã chạy)")
parser.add_argument("--output", default="evaluation/results_finetuned.json")
args = parser.parse_args()

from langchain_huggingface import HuggingFaceEmbeddings

os.environ["USE_HYBRID_RETRIEVAL"] = "true"
os.environ["USE_FINETUNED"] = "true"  # skip custom transformer, dùng fine-tuned e5

FINETUNED_PATH = Path("models/finetuned-embedder")
if not FINETUNED_PATH.exists():
    raise FileNotFoundError("Chưa có fine-tuned model. Chạy app một lần để tự tải từ Hugging Face Hub.")

print("Loading fine-tuned e5 embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=str(FINETUNED_PATH),
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

from src.llm_chain import LLM_MODEL, LLM_PROVIDER
from src.retrieval_hybrid import qa_chain
from src.evaluation_enhanced import RAGEvaluator

evaluator = RAGEvaluator(
    qa_chain=qa_chain,
    embeddings=embeddings,
    test_set_path=args.test_set,
)
evaluator.run_full_evaluation(
    limit=args.limit,
    ragas_limit=args.ragas_limit,
    output_path=args.output,
    config={
        "date": date.today().isoformat(),
        "embedder": "fine-tuned multilingual-e5-base",
        "answer_llm": f"{LLM_PROVIDER}/{LLM_MODEL}",
        "query_rewrite": os.getenv("QUERY_REWRITE", "auto"),
    },
)
