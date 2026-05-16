"""Run evaluation only — skips training steps."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
