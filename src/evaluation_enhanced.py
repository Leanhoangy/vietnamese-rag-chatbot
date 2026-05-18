"""
Enhanced evaluation metrics for RAG system.
Includes retrieval metrics (NDCG, MRR, Recall), answer quality metrics (BLEU, ROUGE),
and RAGAS evaluation.
"""

import json
import os
import re
import numpy as np
from typing import List, Dict, Tuple
import time

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # noqa: F401
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rouge_score import rouge_scorer
import nltk

# Import based on availability
try:
    from nltk.translate.bleu_score import sentence_bleu
    BLEU_AVAILABLE = True
except:
    BLEU_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import pytorch_cos_sim
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

load_dotenv()


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
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.semantic_model = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.semantic_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        """Tokenizer independent of NLTK punkt resources."""
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    
    def bleu_score(self, reference: str, hypothesis: str) -> float:
        """
        Calculate BLEU score
        
        Args:
            reference: Ground truth answer
            hypothesis: Generated answer
            
        Returns:
            BLEU score [0, 1]
        """
        if not BLEU_AVAILABLE:
            return 0.0

        ref_tokens = self._simple_tokenize(reference)
        hyp_tokens = self._simple_tokenize(hypothesis)

        try:
            score = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            return score
        except:
            return 0.0
    
    def rouge_l_score(self, reference: str, hypothesis: str) -> float:
        """Calculate ROUGE-L (longest common subsequence based)"""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return scores['rougeL'].fmeasure
        except:
            return 0.0
    
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

        # RAGAS setup
        self.ragas_model = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        self.ragas_llm = LangchainLLMWrapper(self.ragas_model)

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
            
            try:
                result = self.qa_chain.invoke({"query": case["question"]})
                source_docs = result.get("source_documents", [])
                context = [doc.page_content for doc in source_docs]
                
                questions.append(case["question"])
                answers.append(result["result"])
                contexts.append(context if context else [result["result"]])
                ground_truths.append(self._normalize_ground_truths(case.get("ground_truth", "")))
                retrieved_docs.append(context)
                
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"  Error: {e}")
        
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
    ) -> Dict:
        """Run RAGAS evaluation"""
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
        
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": [truths[0] if truths else "" for truths in ground_truths],
        })
        
        result = evaluate(
            dataset,
            metrics=[self.faithfulness_metric, self.answer_relevancy_metric, self.context_precision_metric, self.context_recall_metric],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
            run_config=RunConfig(max_workers=2, timeout=300)
        )
        
        df = result.to_pandas()
        scores = {
            "faithfulness": self._safe_mean(df['faithfulness'].tolist()),
            "answer_relevancy": self._safe_mean(df['answer_relevancy'].tolist()),
            "context_precision": self._safe_mean(df['context_precision'].tolist()),
            "context_recall": self._safe_mean(df['context_recall'].tolist()),
        }
        
        print("\n=== RAGAS Scores ===")
        for metric, score in scores.items():
            print(f"{metric:.<30} {score:.4f}")
        
        return scores
    
    def evaluate_answer_quality(
        self,
        answers: List[str],
        references: List[str],
    ) -> Dict:
        """Evaluate answer generation quality"""
        print("\n" + "="*60)
        print("Evaluating Answer Quality")
        print("="*60)
        
        bleu_scores = []
        rouge_scores = []
        semantic_scores = []
        
        for hyp, ref in zip(answers, references):
            if ref and hyp:
                bleu_scores.append(self.answer_metrics.bleu_score(ref, hyp))
                rouge_scores.append(self.answer_metrics.rouge_l_score(ref, hyp))
                semantic_scores.append(self.answer_metrics.semantic_similarity(ref, hyp))
        
        scores = {
            "bleu": float(np.mean(bleu_scores)) if bleu_scores else 0.0,
            "rouge_l": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
            "semantic_similarity": float(np.mean(semantic_scores)) if semantic_scores else 0.0,
        }
        
        print("\n=== Answer Quality Scores ===")
        for metric, score in scores.items():
            print(f"{metric:.<30} {score:.4f}")
        
        return scores
    
    def run_full_evaluation(self, limit: int = 5) -> Dict:
        """Run complete evaluation pipeline"""
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
        ragas_scores = self.evaluate_ragas(questions, answers, contexts, ground_truths)
        
        # Answer quality evaluation
        answer_scores = self.evaluate_answer_quality(
            answers,
            [truths[0] if truths else "" for truths in ground_truths],
        )
        
        # Combine results
        all_scores = {
            "retrieval": retrieval_scores,
            "ragas": ragas_scores,
            "answer_quality": answer_scores,
        }
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(json.dumps(all_scores, indent=2))
        
        # Save results
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=2, ensure_ascii=False)
        print("\nResults saved to: evaluation_results.json")
        
        return all_scores


if __name__ == "__main__":
    # This module is intended to be used from train.py or tests.
    print("Enhanced evaluation module loaded. Use RAGEvaluator class directly.")
