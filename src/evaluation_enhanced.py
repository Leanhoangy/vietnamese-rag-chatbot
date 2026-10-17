"""
Enhanced evaluation metrics for RAG system.
Includes retrieval metrics (NDCG, MRR, Recall), answer quality metrics (Semantic Similarity),
and RAGAS evaluation.
"""

import json
import os
import re
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import time

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # noqa: F401
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import pytorch_cos_sim
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

load_dotenv()

# Free tier giới hạn request/phút → gặp 429 thì chờ rồi thử lại
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT_SECONDS = 30


def _is_rate_limit(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "rate_limit" in message or "resource_exhausted" in message


class RetrievalMetrics:
    """Calculate retrieval-specific metrics"""
    
    @staticmethod
    def ndcg_at_k(relevant_indices: List[int], rank: np.ndarray, k: int = 5) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain)
        
        Args:
            relevant_indices: Indices of relevant documents
            rank: Ranked indices from retriever
            k: Cutoff rank
            
        Returns:
            NDCG score [0, 1]
        """
        # Ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_indices), k)))
        
        if ideal_dcg == 0:
            return 0.0
        
        # Actual DCG
        dcg = 0.0
        for i, idx in enumerate(rank[:k]):
            if idx in relevant_indices:
                dcg += 1.0 / np.log2(i + 2)
        
        return dcg / ideal_dcg
    
    @staticmethod
    def mrr_at_k(relevant_indices: List[int], rank: np.ndarray, k: int = 10) -> float:
        """
        Calculate MRR@k (Mean Reciprocal Rank)
        
        Returns:
            Reciprocal rank of first relevant item, 0 if not in top-k
        """
        for i, idx in enumerate(rank[:k]):
            if idx in relevant_indices:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def recall_at_k(relevant_indices: List[int], rank: np.ndarray, k: int = 5) -> float:
        """Calculate Recall@k (how many relevant items are in top-k)"""
        if not relevant_indices:
            return 0.0
        retrieved = set(rank[:k])
        relevant = set(relevant_indices)
        return len(retrieved & relevant) / len(relevant)
    
    @staticmethod
    def precision_at_k(relevant_indices: List[int], rank: np.ndarray, k: int = 5) -> float:
        """Calculate Precision@k"""
        if k == 0:
            return 0.0
        retrieved = set(rank[:k])
        relevant = set(relevant_indices)
        return len(retrieved & relevant) / k


class AnswerQualityMetrics:
    """Calculate answer generation quality metrics"""

    def __init__(self):
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.semantic_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

    def semantic_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Calculate semantic similarity using sentence-transformers
        
        Returns:
            Cosine similarity [0, 1]
        """
        try:
            if self.semantic_model is None:
                return 0.0
            emb_ref = self.semantic_model.encode(reference, convert_to_tensor=True)
            emb_hyp = self.semantic_model.encode(hypothesis, convert_to_tensor=True)
            score = pytorch_cos_sim(emb_ref, emb_hyp).item()
            return float(score)
        except:
            return 0.0


class RAGEvaluator:
    """Comprehensive RAG evaluation"""
    
    def __init__(self, qa_chain, embeddings, test_set_path: str = "test_set.json"):
        """
        Initialize evaluator
        
        Args:
            qa_chain: LangChain QA chain
            embeddings: Embedding model (HuggingFace wrapper)
            test_set_path: Path to test set JSON
        """
        self.qa_chain = qa_chain
        self.embeddings = embeddings
        self.test_set_path = test_set_path
        
        # Initialize metrics calculators
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerQualityMetrics()

        # Giám khảo RAGAS: mặc định cùng provider với chatbot. RAGAS gọi LLM ~10 lần/câu (phần lớn quota),
        # nên EVAL_JUDGE_PROVIDER=ollama cho phép chấm bằng model local mà vẫn để Gemini sinh câu trả lời.
        from src.llm_chain import LLM_PROVIDER, make_llm, resolve_llm

        # Free tier Gemini Flash-Lite: 15 request/phút; mỗi câu ~2 request (viết lại câu hỏi + trả lời) → ~8 giây/câu
        self.request_interval = float(os.getenv("EVAL_REQUEST_INTERVAL", "8" if LLM_PROVIDER == "gemini" else "0.5"))
        self.judge_provider, self.judge_model = resolve_llm(
            os.getenv("EVAL_JUDGE_PROVIDER") or None,
            os.getenv("EVAL_JUDGE_MODEL") or None,
        )
        print(f"RAGAS judge: {self.judge_provider} / {self.judge_model}")
        self.ragas_model = make_llm(provider=self.judge_provider, model_name=self.judge_model)
        self.ragas_llm = LangchainLLMWrapper(self.ragas_model)
        self.ragas_valid_samples: Dict[str, int] = {}

        # RAGAS requires a HuggingFace embedding with a string model name.
        # Always use e5-base regardless of what the QA chain uses.
        from langchain_huggingface import HuggingFaceEmbeddings
        ragas_embed = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.ragas_embeddings = LangchainEmbeddingsWrapper(ragas_embed)

        faithfulness.llm = self.ragas_llm
        answer_relevancy.llm = self.ragas_llm
        answer_relevancy.embeddings = self.ragas_embeddings
        answer_relevancy.strictness = 1
        context_precision.llm = self.ragas_llm
        context_recall.llm = self.ragas_llm
        self.faithfulness_metric = faithfulness
        self.answer_relevancy_metric = answer_relevancy
        self.context_precision_metric = context_precision
        self.context_recall_metric = context_recall
    
    def load_test_set(self) -> List[Dict]:
        """Load test set"""
        with open(self.test_set_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def _normalize_ground_truths(ground_truth) -> List[str]:
        if isinstance(ground_truth, list):
            return [item for item in ground_truth if isinstance(item, str) and item.strip()]
        if isinstance(ground_truth, str) and ground_truth.strip():
            return [ground_truth]
        return []

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _is_relevant(doc_text: str, ground_truths: List[str]) -> bool:
        doc_text_norm = RAGEvaluator._normalize_text(doc_text)
        doc_tokens = set(re.findall(r"\w+", doc_text_norm, flags=re.UNICODE))

        for gt in ground_truths:
            gt_norm = RAGEvaluator._normalize_text(gt)
            if not gt_norm:
                continue

            if gt_norm in doc_text_norm or doc_text_norm in gt_norm:
                return True

            gt_tokens = set(re.findall(r"\w+", gt_norm, flags=re.UNICODE))
            if not gt_tokens or not doc_tokens:
                continue

            overlap = doc_tokens & gt_tokens
            overlap_gt = len(overlap) / len(gt_tokens)
            overlap_doc = len(overlap) / len(doc_tokens)

            if len(overlap) >= 8 or overlap_gt >= 0.3 or overlap_doc >= 0.3:
                return True
        return False

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        if not values:
            return 0.0

        numeric_values = np.array(values, dtype=float)
        if np.isnan(numeric_values).all():
            return 0.0

        return float(np.nanmean(numeric_values))

    def run_qa_pipeline(
        self,
        test_cases: List[Dict],
        limit: int = None,
    ) -> Tuple[List, List, List, List, List[List[str]]]:
        """
        Run QA pipeline on test cases
        
        Returns:
            (questions, answers, contexts, ground_truths, retrieved_docs)
        """
        questions, answers, contexts, ground_truths, retrieved_docs = [], [], [], [], []
        
        test_cases = test_cases[:limit] if limit else test_cases
        
        for i, case in enumerate(test_cases):
            print(f"[{i+1}/{len(test_cases)}] {case['question'][:60]}...")

            for attempt in range(RATE_LIMIT_RETRIES + 1):
                try:
                    result = self.qa_chain.invoke({"query": case["question"]})
                    source_docs = result.get("source_documents", [])
                    context = [doc.page_content for doc in source_docs]

                    questions.append(case["question"])
                    answers.append(result["result"])
                    contexts.append(context if context else [result["result"]])
                    ground_truths.append(self._normalize_ground_truths(case.get("ground_truth", "")))
                    retrieved_docs.append(context)

                    time.sleep(self.request_interval)
                    break
                except Exception as e:
                    if _is_rate_limit(e) and attempt < RATE_LIMIT_RETRIES:
                        print(f"  [Rate limit] Waiting {RATE_LIMIT_WAIT_SECONDS}s, retrying...")
                        time.sleep(RATE_LIMIT_WAIT_SECONDS)
                    else:
                        print(f"  Error: {e}")
                        break
        
        return questions, answers, contexts, ground_truths, retrieved_docs

    def evaluate_retrieval(
        self,
        retrieved_docs: List[List[str]],
        ground_truths: List[List[str]],
    ) -> Dict:
        """Evaluate ranking quality of retrieved contexts."""
        print("\n" + "="*60)
        print("Evaluating Retrieval Quality")
        print("="*60)

        ndcg_scores = []
        mrr_scores = []
        recall_scores = []
        precision_scores = []

        for docs, truths in zip(retrieved_docs, ground_truths):
            relevant_indices = []
            rank = np.arange(len(docs))

            for idx, doc_text in enumerate(docs):
                if self._is_relevant(doc_text, truths):
                    relevant_indices.append(idx)

            ndcg_scores.append(self.retrieval_metrics.ndcg_at_k(relevant_indices, rank, k=7))
            mrr_scores.append(self.retrieval_metrics.mrr_at_k(relevant_indices, rank, k=10))
            recall_scores.append(self.retrieval_metrics.recall_at_k(relevant_indices, rank, k=7))
            precision_scores.append(self.retrieval_metrics.precision_at_k(relevant_indices, rank, k=7))

        scores = {
            "ndcg@7": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "mrr@10": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            "recall@7": float(np.mean(recall_scores)) if recall_scores else 0.0,
            "precision@7": float(np.mean(precision_scores)) if precision_scores else 0.0,
        }

        print("\n=== Retrieval Scores ===")
        for metric, score in scores.items():
            print(f"{metric:.<30} {score:.4f}")

        return scores
    
    def evaluate_ragas(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
        ragas_limit: int | None = 50,
    ) -> Dict:
        """Run RAGAS evaluation on the first ragas_limit samples (None = all)."""
        print("\n" + "="*60)
        print("Running RAGAS Evaluation")
        print("="*60)

        if not questions or not answers or not contexts:
            print("No successful QA samples available. Skipping RAGAS evaluation.")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
        
        questions = questions[:ragas_limit]
        answers = answers[:ragas_limit]
        contexts = contexts[:ragas_limit]
        ground_truths = ground_truths[:ragas_limit]
        print(f"Evaluating {len(questions)} samples with RAGAS...")

        # Chạy từng sample — 429 thì chờ rồi retry sample đó, không restart từ đầu
        all_scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "context_recall": []}
        ground_truth_list = [truths[0] if truths else "" for truths in ground_truths]

        for i, (q, a, c, gt) in enumerate(zip(questions, answers, contexts, ground_truth_list)):
            print(f"  [RAGAS {i+1}/{len(questions)}]", end=" ", flush=True)
            single = Dataset.from_dict({"question": [q], "answer": [a], "contexts": [c], "ground_truth": [gt]})
            success = False
            for attempt in range(RATE_LIMIT_RETRIES + 1):
                try:
                    res = evaluate(
                        single,
                        metrics=[self.faithfulness_metric, self.answer_relevancy_metric,
                                 self.context_precision_metric, self.context_recall_metric],
                        llm=self.ragas_llm,
                        embeddings=self.ragas_embeddings,
                        run_config=RunConfig(max_workers=1, timeout=600),  # giám khảo local chậm hơn API
                    )
                    df = res.to_pandas()
                    for metric in all_scores:
                        val = df[metric].iloc[0] if metric in df.columns else float("nan")
                        all_scores[metric].append(float(val) if val == val else float("nan"))
                    print("✓")
                    success = True
                    break
                except Exception as e:
                    if _is_rate_limit(e) and attempt < RATE_LIMIT_RETRIES:
                        print(f"429→wait {RATE_LIMIT_WAIT_SECONDS}s", end=" ", flush=True)
                        time.sleep(RATE_LIMIT_WAIT_SECONDS)
                    else:
                        print(f"skip({e})")
                        for metric in all_scores:
                            all_scores[metric].append(float("nan"))
                        success = True
                        break
            if not success:
                for metric in all_scores:
                    all_scores[metric].append(float("nan"))

        # Giám khảo có thể trả JSON lỗi → NaN; ghi lại số mẫu chấm được để biết điểm đáng tin tới đâu
        self.ragas_valid_samples = {metric: int(np.sum(~np.isnan(vals))) for metric, vals in all_scores.items()}
        scores = {
            metric: float(np.nanmean(vals)) if self.ragas_valid_samples[metric] else 0.0
            for metric, vals in all_scores.items()
        }

        print("\n=== RAGAS Scores ===")
        for metric, score in scores.items():
            print(f"{metric:.<30} {score:.4f}  ({self.ragas_valid_samples[metric]}/{len(questions)} samples)")

        return scores
    
    def evaluate_answer_quality(
        self,
        answers: List[str],
        references: List[str],
    ) -> Dict:
        """Evaluate answer semantic similarity vs ground truth"""
        print("\n" + "="*60)
        print("Evaluating Answer Quality")
        print("="*60)

        semantic_scores = []
        for hyp, ref in zip(answers, references):
            if ref and hyp:
                semantic_scores.append(self.answer_metrics.semantic_similarity(ref, hyp))

        scores = {
            "semantic_similarity": float(np.mean(semantic_scores)) if semantic_scores else 0.0,
        }

        print("\n=== Answer Quality Scores ===")
        for metric, score in scores.items():
            print(f"{metric:.<30} {score:.4f}")

        return scores
    
    def run_full_evaluation(
        self,
        limit: int | None = 5,
        ragas_limit: int | None = 50,
        output_path: str = "evaluation_results.json",
        config: Dict | None = None,
    ) -> Dict:
        """Run complete evaluation pipeline and save scores together with the run configuration."""
        print("Loading test set...")
        test_cases = self.load_test_set()
        
        print("Running QA pipeline...")
        questions, answers, contexts, ground_truths, retrieved_docs = self.run_qa_pipeline(
            test_cases,
            limit=limit,
        )

        if not questions:
            raise RuntimeError(
                "QA pipeline did not produce any successful samples. "
                "Check retrieval and model configuration before evaluation."
            )
        
        retrieval_scores = self.evaluate_retrieval(retrieved_docs, ground_truths)
        
        # RAGAS evaluation
        ragas_scores = self.evaluate_ragas(questions, answers, contexts, ground_truths, ragas_limit=ragas_limit)
        
        # Answer quality evaluation
        answer_scores = self.evaluate_answer_quality(
            answers,
            [truths[0] if truths else "" for truths in ground_truths],
        )
        
        # Combine results
        all_scores = {
            "config": {
                "test_set": self.test_set_path,
                "questions_evaluated": len(questions),
                "judge_llm": f"{self.judge_provider}/{self.judge_model}",
                **(config or {}),
            },
            "retrieval": retrieval_scores,
            "ragas": ragas_scores,
            "ragas_valid_samples": self.ragas_valid_samples,
            "answer_quality": answer_scores,
        }

        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(json.dumps(all_scores, indent=2, ensure_ascii=False))

        # Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

        return all_scores


if __name__ == "__main__":
    # This module is intended to be used from train.py or tests.
    print("Enhanced evaluation module loaded. Use RAGEvaluator class directly.")
