"""Run evaluation only — skips training steps."""

import os

os.environ["USE_HYBRID_RETRIEVAL"] = "true"

from src.evaluation_enhanced import RAGEvaluator
from src.embedder import embeddings
from src.retrieval_hybrid import qa_chain

evaluator = RAGEvaluator(
    qa_chain=qa_chain,
    embeddings=embeddings,
    test_set_path="test_set.json",
)
evaluator.run_full_evaluation(limit=10)
