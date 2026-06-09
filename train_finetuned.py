"""
Main training pipeline: Fine-tune embedding model
This is the entry point for training the deep learning component
"""

import os
import subprocess
import sys
from pathlib import Path
import traceback


def install_dependencies():
    command = [
        sys.executable, "-m", "pip", "install", "-q",
        "accelerate>=1.1.0", "rank-bm25", "rouge-score", "scikit-learn", "nltk",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        print("✓ Dependencies installed")
    else:
        print("⚠ Dependency installation had issues. Continuing with current environment...")


def prepare_training_data():
    print("\n" + "="*70)
    print("STEP 1: Preparing Training Data")
    print("="*70)

    dataset_path = Path("legal_triplets_dataset")
    if dataset_path.exists():
        from datasets import load_from_disk
        dataset = load_from_disk(str(dataset_path))
        print(f"\n✓ Dataset already exists — skipping regeneration")
        print(f"  - Triplets: {len(dataset)}")
        return True

    os.environ["USE_FINETUNED"] = "true"
    from src.data_preparation import LegalDataPreparator
    preparator = LegalDataPreparator(test_set_path="train_set.json", k_hard_negatives=3)
    dataset = preparator.prepare_all(save_to_disk=True)

    if dataset:
        print(f"\n✓ Dataset prepared: {len(dataset)} triplets")
        return True
    print("✗ Failed to prepare dataset")
    return False


def finetune_embedding_model():
    print("\n" + "="*70)
    print("STEP 2: Fine-tuning Embedding Model")
    print("="*70)

    from src.finetuned_e5.trainer import run_finetuning_pipeline

    if not Path("legal_triplets_dataset").exists():
        print("✗ Dataset not found. Run data preparation first!")
        return False

    try:
        tuner = run_finetuning_pipeline(dataset_path="legal_triplets_dataset", epochs=15, batch_size=32)
        print(f"✓ Fine-tuning completed! Model saved to: {tuner.output_dir}")
        return True
    except Exception as e:
        print(f"✗ Fine-tuning failed: {e}")
        traceback.print_exc()
        return False


def update_embedder():
    print("\n" + "="*70)
    print("STEP 3: Verifying Embedder")
    print("="*70)

    if Path("models/finetuned-embedder").exists():
        print("✓ Fine-tuned model found at: models/finetuned-embedder")
        return True
    print("✗ Fine-tuned model directory not found")
    return False


def main():
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Vietnamese Legal RAG - Training Pipeline" + " "*13 + "║")
    print("╚" + "="*68 + "╝")

    install_dependencies()

    if not prepare_training_data():
        print("\n✗ Training pipeline failed at Step 1")
        sys.exit(1)

    if not finetune_embedding_model():
        print("\n✗ Training pipeline failed at Step 2")
        sys.exit(1)

    if not update_embedder():
        print("\n✗ Training pipeline failed at Step 3")
        sys.exit(1)

    print("\n✓ Training pipeline completed successfully!")
    print("  - Fine-tuned embedding model: models/finetuned-embedder/")
    print("  - Chạy evaluation: python evaluate_finetuned.py")


if __name__ == "__main__":
    main()
