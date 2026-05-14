"""
Fine-tuning multilingual-e5-base on legal domain Q&A pairs.
Uses SentenceTransformers with TripletLoss and hard negative mining.
"""

import inspect
import os
import numpy as np
from importlib import import_module
from pathlib import Path
from typing import Any, List, Tuple

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk

from dotenv import load_dotenv

load_dotenv()


def _default_training_device() -> str:
    """Prefer stability on Apple Silicon by defaulting to CPU unless overridden."""
    device_override = os.getenv("FINETUNE_DEVICE")
    if device_override:
        return device_override

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"

    return "cpu"


def _load_ir_evaluator():
    """Import the evaluator lazily to avoid version-specific import issues."""
    module = import_module("sentence_transformers.evaluation")
    return module.InformationRetrievalEvaluator


class EmbeddingFineTuner:
    """Fine-tune multilingual-e5-base for legal domain"""
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        output_dir: str = "models/finetuned-embedder",
        device: str = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device or _default_training_device()
        
        print(f"Device: {self.device}")
        
        # Load base model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Loaded model with {self.model_params:,} parameters")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare_triplet_data(self, dataset: Dataset) -> List[InputExample]:
        """Convert dataset to InputExample for SentenceTransformers"""
        examples = []
        
        for sample in dataset:
            example = InputExample(
                texts=[sample["anchor"], sample["positive"], sample["negative"]]
            )
            examples.append(example)
        
        print(f"Prepared {len(examples)} training examples")
        return examples

    def prepare_pair_data(self, dataset: Dataset) -> List[InputExample]:
        """Convert dataset to anchor-positive pairs for multiple negatives loss."""
        examples = []

        for sample in dataset:
            examples.append(InputExample(texts=[sample["anchor"], sample["positive"]]))

        print(f"Prepared {len(examples)} pair examples")
        return examples
    
    def create_triplet_loss(self, model, margin: float = 0.5) -> losses.TripletLoss:
        """Create triplet loss while staying compatible across library versions."""
        triplet_signature = inspect.signature(losses.TripletLoss.__init__)

        if "triplet_margin" in triplet_signature.parameters:
            return losses.TripletLoss(model=model, triplet_margin=margin)

        return losses.TripletLoss(model=model, margin=margin)
    
    def create_multiple_negatives_loss(self, model) -> losses.MultipleNegativesRankingLoss:
        """Create in-batch negatives loss (more efficient)"""
        return losses.MultipleNegativesRankingLoss(model=model)
    
    def create_dataloader(
        self,
        examples: List[InputExample],
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create DataLoader for training"""
        return DataLoader(
            examples,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=self.device == "cuda",
        )
    
    def create_evaluator(
        self,
        queries: List[str],
        corpus: dict,
        relevant_docs: dict,
    ) -> Any:
        """
        Create evaluator for validation during training
        
        Args:
            queries: List of query strings
            corpus: Dict {doc_id: doc_content}
            relevant_docs: Dict {query_id: [relevant_doc_ids]}
        """
        information_retrieval_evaluator = _load_ir_evaluator()
        evaluator = information_retrieval_evaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            main_similarity="cosine",
        )
        return evaluator
    
    def fine_tune(
        self,
        train_dataloader: DataLoader,
        evaluator=None,
        epochs: int = 3,
        warmup_steps: int = 100,
        loss_type: str = "triplet",
        margin: float = 0.5,
        learning_rate: float = 2e-5,
    ):
        """
        Fine-tune the embedding model
        
        Args:
            train_dataloader: DataLoader with training examples
            evaluator: Optional evaluator for validation
            epochs: Number of training epochs
            warmup_steps: Warmup steps for scheduler
            loss_type: "triplet" or "multiple_negatives" 
            margin: Margin for triplet loss
            learning_rate: Learning rate
        """
        
        # Choose loss function
        if loss_type == "triplet":
            loss = self.create_triplet_loss(self.model, margin=margin)
        elif loss_type == "multiple_negatives":
            loss = self.create_multiple_negatives_loss(self.model)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        print(f"\n=== Starting Fine-tuning ===")
        print(f"Model: {self.model_name}")
        print(f"Loss: {loss_type} (margin={margin if loss_type=='triplet' else 'N/A'})")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Batch size: {train_dataloader.batch_size}")
        print(f"Total steps: {len(train_dataloader) * epochs}")
        print("=" * 50)

        if self.device == "cpu":
            # Force Hugging Face Trainer / Accelerate to stay on CPU.
            os.environ["ACCELERATE_USE_CPU"] = "true"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif "ACCELERATE_USE_CPU" in os.environ:
            del os.environ["ACCELERATE_USE_CPU"]
        
        has_evaluator = evaluator is not None
        train_objectives = [(train_dataloader, loss)]

        # Prefer the legacy training loop to avoid Trainer/Accelerate device issues on Mac.
        if hasattr(self.model, "old_fit"):
            self.model.old_fit(
                train_objectives=train_objectives,
                evaluator=evaluator,
                epochs=epochs,
                warmup_steps=warmup_steps,
                output_path=self.output_dir,
                save_best_model=has_evaluator,
                show_progress_bar=True,
                optimizer_params={"lr": learning_rate},
                weight_decay=1e-6,
                checkpoint_path=None,
                checkpoint_save_steps=0,
                checkpoint_save_total_limit=0,
            )
        else:
            fit_signature = inspect.signature(self.model.fit)
            fit_kwargs = {
                "train_objectives": train_objectives,
                "evaluator": evaluator,
                "epochs": epochs,
                "evaluation_steps": len(train_dataloader) if has_evaluator else 0,
                "warmup_steps": warmup_steps,
                "output_path": self.output_dir,
                "save_best_model": has_evaluator,
                "show_progress_bar": True,
                "optimizer_params": {
                    "lr": learning_rate,
                },
            }

            if "weight_decay" in fit_signature.parameters:
                fit_kwargs["weight_decay"] = 1e-6

            if "checkpoint_path" in fit_signature.parameters:
                fit_kwargs["checkpoint_path"] = None

            if "checkpoint_save_steps" in fit_signature.parameters:
                fit_kwargs["checkpoint_save_steps"] = -1

            if "checkpoint_save_total_limit" in fit_signature.parameters:
                fit_kwargs["checkpoint_save_total_limit"] = 0

            self.model.fit(**fit_kwargs)
        
        print(f"\n✓ Fine-tuning complete!")
        print(f"Model saved to: {self.output_dir}")
    
    def evaluate_before_after(
        self,
        test_queries: List[str],
        corpus_docs: List[str],
        query_corpus_pairs: List[Tuple[int, int]],  # [(query_idx, corpus_idx)]
    ):
        """
        Compare embedding quality before and after fine-tuning
        
        Args:
            test_queries: List of test queries
            corpus_docs: List of corpus documents
            query_corpus_pairs: List of (query_idx, corpus_idx) indicating relevance
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("\n=== Evaluating Embedding Quality ===")
        
        # Encode with original model
        print("Encoding with original model...")
        original_model = SentenceTransformer(self.model_name, device=self.device)
        original_query_embeddings = original_model.encode(test_queries)
        original_corpus_embeddings = original_model.encode(corpus_docs)
        
        # Encode with fine-tuned model
        print("Encoding with fine-tuned model...")
        finetuned_query_embeddings = self.model.encode(test_queries)
        finetuned_corpus_embeddings = self.model.encode(corpus_docs)
        
        # Calculate metrics
        def calc_metrics(query_embs, corpus_embs, pairs):
            sim_matrix = cosine_similarity(query_embs, corpus_embs)
            
            mrr = 0
            recall_at_5 = 0
            recall_at_10 = 0
            
            for q_idx, c_idx in pairs:
                scores = sim_matrix[q_idx]
                rank = np.argsort(scores)[::-1]
                true_idx_in_rank = np.where(rank == c_idx)[0]
                
                if len(true_idx_in_rank) > 0:
                    position = true_idx_in_rank[0]
                    mrr += 1 / (position + 1)
                    if position < 5:
                        recall_at_5 += 1
                    if position < 10:
                        recall_at_10 += 1
            
            n_pairs = len(pairs)
            return {
                "MRR": mrr / n_pairs,
                "Recall@5": recall_at_5 / n_pairs,
                "Recall@10": recall_at_10 / n_pairs,
            }
        
        original_metrics = calc_metrics(original_query_embeddings, original_corpus_embeddings, query_corpus_pairs)
        finetuned_metrics = calc_metrics(finetuned_query_embeddings, finetuned_corpus_embeddings, query_corpus_pairs)
        
        print(f"\nOriginal Model:")
        for metric, value in original_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nFine-tuned Model:")
        for metric, value in finetuned_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nImprovement:")
        for metric in original_metrics.keys():
            baseline = original_metrics[metric]
            if baseline == 0:
                print(f"  {metric}: N/A (baseline = 0)")
                continue
            improvement = (finetuned_metrics[metric] - baseline) / baseline * 100
            print(f"  {metric}: {improvement:+.2f}%")
        
        return {
            "original": original_metrics,
            "finetuned": finetuned_metrics,
        }
    
    def load_finetuned_model(self):
        """Load the fine-tuned model"""
        return SentenceTransformer(self.output_dir, device=self.device)


def run_finetuning_pipeline(
    dataset_path: str = "legal_triplets_dataset",
    epochs: int = 3,
    batch_size: int = 16,
    loss_type: str = "triplet",
):
    """End-to-end fine-tuning pipeline"""

    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} training examples")

    # Initialize fine-tuner
    tuner = EmbeddingFineTuner(output_dir="models/finetuned-embedder")

    # Prepare training data
    if loss_type == "triplet":
        examples = tuner.prepare_triplet_data(dataset)
    elif loss_type == "multiple_negatives":
        examples = tuner.prepare_pair_data(dataset)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    train_dataloader = tuner.create_dataloader(examples, batch_size=batch_size)

    # warmup = 10% of total steps, minimum 1
    total_steps = len(train_dataloader) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    print(f"Total training steps: {total_steps}, warmup steps: {warmup_steps}")

    # Fine-tune
    tuner.fine_tune(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        loss_type=loss_type,
        margin=0.5,
        learning_rate=2e-5,
    )
    
    print(f"\n✓ Fine-tuning complete! Model saved to: {tuner.output_dir}")
    return tuner


if __name__ == "__main__":
    # Check if dataset exists
    if not Path("legal_triplets_dataset").exists():
        print("Dataset not found! Run data_preparation.py first:")
        print("  python src/data_preparation.py")
        exit(1)
    
    # Run fine-tuning
    tuner = run_finetuning_pipeline(
        dataset_path="legal_triplets_dataset",
        epochs=3,
        batch_size=16,
    )
