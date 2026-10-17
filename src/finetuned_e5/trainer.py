"""
Fine-tuning multilingual-e5-base on legal domain Q&A pairs.
Uses SentenceTransformers with TripletLoss and hard negative mining.
"""

import inspect
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk

from dotenv import load_dotenv

load_dotenv()


class TripletLossEvaluator:
    """Tính test loss sau mỗi epoch, lưu vào JSON."""

    def __init__(self, test_examples: List[InputExample], margin: float = 0.5,
                 log_path: str = "loss_log_test_finetuned.json"):
        self.test_examples = test_examples
        self.margin = margin
        self.log_path = log_path
        self.log = []
        self._criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def __call__(self, model: SentenceTransformer, output_path=None, epoch: int = -1, steps: int = -1, **_):
        anchors   = [ex.texts[0] for ex in self.test_examples]
        positives = [ex.texts[1] for ex in self.test_examples]
        negatives = [ex.texts[2] for ex in self.test_examples]

        with torch.no_grad():
            a = model.encode(anchors,   batch_size=64, convert_to_tensor=True, show_progress_bar=False)
            p = model.encode(positives, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
            n = model.encode(negatives, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
            loss = self._criterion(a.cpu(), p.cpu(), n.cpu()).item()

        self.log.append({"epoch": epoch, "loss": loss})
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"  [Test Loss] Epoch {epoch}: {loss:.4f}")
        return -loss  # SentenceTransformers: higher = better


def _default_training_device() -> str:
    device_override = os.getenv("FINETUNE_DEVICE")
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Loaded model with {self.model_params:,} parameters")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def prepare_triplet_data(self, dataset: Dataset) -> List[InputExample]:
        examples = []
        for sample in dataset:
            examples.append(InputExample(
                texts=[sample["anchor"], sample["positive"], sample["negative"]]
            ))
        print(f"Prepared {len(examples)} training examples")
        return examples

    def create_triplet_loss(self, model, margin: float = 0.5) -> losses.TripletLoss:
        triplet_signature = inspect.signature(losses.TripletLoss.__init__)
        if "triplet_margin" in triplet_signature.parameters:
            return losses.TripletLoss(model=model, triplet_margin=margin)
        return losses.TripletLoss(model=model, margin=margin)

    def create_dataloader(
        self,
        examples: List[InputExample],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        return DataLoader(
            examples,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=self.device == "cuda",
        )

    def fine_tune(
        self,
        train_dataloader: DataLoader,
        epochs: int = 15,
        warmup_steps: int = 100,
        margin: float = 0.5,
        learning_rate: float = 2e-5,
        evaluator: TripletLossEvaluator = None,
    ):
        loss = self.create_triplet_loss(self.model, margin=margin)

        print(f"\n=== Starting Fine-tuning ===")
        print(f"Model: {self.model_name}")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Batch size: {train_dataloader.batch_size}")
        print(f"Total steps: {len(train_dataloader) * epochs}")
        print("=" * 50)

        if self.device == "cpu":
            os.environ["ACCELERATE_USE_CPU"] = "true"

        train_objectives = [(train_dataloader, loss)]

        if hasattr(self.model, "old_fit"):
            self.model.old_fit(
                train_objectives=train_objectives,
                epochs=epochs,
                warmup_steps=warmup_steps,
                output_path=self.output_dir,
                save_best_model=False,
                show_progress_bar=True,
                optimizer_params={"lr": learning_rate},
                weight_decay=1e-6,
                checkpoint_path=None,
                checkpoint_save_steps=0,
                checkpoint_save_total_limit=0,
                evaluator=evaluator,
                evaluation_steps=0,
            )
        else:
            fit_signature = inspect.signature(self.model.fit)
            fit_kwargs = {
                "train_objectives": train_objectives,
                "epochs": epochs,
                "evaluation_steps": 0,
                "warmup_steps": warmup_steps,
                "output_path": self.output_dir,
                "save_best_model": False,
                "show_progress_bar": True,
                "optimizer_params": {"lr": learning_rate},
            }
            if evaluator is not None and "evaluator" in fit_signature.parameters:
                fit_kwargs["evaluator"] = evaluator
            if "weight_decay" in fit_signature.parameters:
                fit_kwargs["weight_decay"] = 1e-6
            if "checkpoint_path" in fit_signature.parameters:
                fit_kwargs["checkpoint_path"] = None
            if "checkpoint_save_steps" in fit_signature.parameters:
                fit_kwargs["checkpoint_save_steps"] = -1
            if "checkpoint_save_total_limit" in fit_signature.parameters:
                fit_kwargs["checkpoint_save_total_limit"] = 0
            self.model.fit(**fit_kwargs)

        print(f"\n✓ Fine-tuning complete! Model saved to: {self.output_dir}")


def run_finetuning_pipeline(
    dataset_path: str = "legal_triplets_dataset",
    test_dataset_path: str = "test_triplets_dataset",
    epochs: int = 15,
    batch_size: int = 32,
):
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} training examples")

    tuner = EmbeddingFineTuner(output_dir="models/finetuned-embedder")
    examples = tuner.prepare_triplet_data(dataset)
    train_dataloader = tuner.create_dataloader(examples, batch_size=batch_size)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    print(f"Total training steps: {total_steps}, warmup steps: {warmup_steps}")

    evaluator = None
    if Path(test_dataset_path).exists():
        test_dataset = load_from_disk(test_dataset_path)
        test_examples = tuner.prepare_triplet_data(test_dataset)
        evaluator = TripletLossEvaluator(test_examples, margin=0.5)
        print(f"Test evaluator ready — {len(test_examples)} triplets")

    tuner.fine_tune(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        margin=0.5,
        learning_rate=2e-5,
        evaluator=evaluator,
    )

    return tuner
