"""Evaluate hệ thống dùng fine-tuned e5 (bỏ qua custom transformer)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings

os.environ["USE_HYBRID_RETRIEVAL"] = "true"
os.environ["USE_FINETUNED"] = "true"  # skip custom transformer, dùng fine-tuned e5
os.environ["EVAL_LLM_MODEL"] = "llama-3.1-8b-instant"  # dùng 8b cho QA pipeline khi evaluation

FINETUNED_PATH = Path("models/finetuned-embedder")
if not FINETUNED_PATH.exists():
    raise FileNotFoundError("Chưa có fine-tuned model. Chạy train.py trước.")

print("Loading fine-tuned e5 embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=str(FINETUNED_PATH),
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

from src.retrieval_hybrid import qa_chain
from src.evaluation_enhanced import RAGEvaluator

evaluator = RAGEvaluator(
    qa_chain=qa_chain,
    embeddings=embeddings,
    test_set_path="test_set.json",
)
evaluator.run_full_evaluation(limit=150)
