"""
Main training pipeline: Fine-tune embedding model + evaluate
This is the entry point for training the deep learning component
"""

import os
import subprocess
import sys
from pathlib import Path
import traceback

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "accelerate>=1.1.0",
        "rank-bm25",
        "rouge-score",
        "scikit-learn",
        "nltk",
    ]

    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        print("✓ Dependencies installed")
    else:
        print("⚠ Dependency installation had issues. Continuing with current environment...")


def prepare_training_data():
    """Step 1: Prepare training data for fine-tuning"""
    print("\n" + "="*70)
    print("STEP 1: Preparing Training Data")
    print("="*70)

    dataset_path = Path("legal_triplets_dataset")
    if dataset_path.exists():
        from datasets import load_from_disk
        dataset = load_from_disk(str(dataset_path))
        print(f"\n✓ Dataset already exists — skipping regeneration")
        print(f"  - Triplets: {len(dataset)}")
        print(f"  - Path: {dataset_path}/")
        return True

    from src.data_preparation import LegalDataPreparator

    print("\nInitializing data preparation...")
    preparator = LegalDataPreparator(
        test_set_path="test_set.json",
        k_hard_negatives=3,
    )

    print("Loading and preparing data...")
    dataset = preparator.prepare_all(save_to_disk=True)

    if dataset:
        print(f"\n✓ Dataset prepared successfully!")
        print(f"  - Triplets: {len(dataset)}")
        print(f"  - Saved to: {dataset_path}/")
        return True
    else:
        print("✗ Failed to prepare dataset")
        return False


def finetune_embedding_model():
    """Step 2: Fine-tune the embedding model"""
    print("\n" + "="*70)
    print("STEP 2: Fine-tuning Embedding Model")
    print("="*70)
    
    from pathlib import Path
    from src.fine_tuner import run_finetuning_pipeline
    
    dataset_path = "legal_triplets_dataset"
    if not Path(dataset_path).exists():
        print(f"✗ Dataset not found at {dataset_path}")
        print("  Run data preparation first!")
        return False
    
    print("\nStarting fine-tuning...")
    print("-" * 70)
    
    try:
        tuner = run_finetuning_pipeline(
            dataset_path=dataset_path,
            epochs=5,
            batch_size=16,
        )
        print("-" * 70)
        print("✓ Fine-tuning completed!")
        print(f"  - Model saved to: {tuner.output_dir}")
        return True
    except Exception as e:
        print(f"✗ Fine-tuning failed: {e}")
        traceback.print_exc()
        return False


def update_embedder():
    """Step 3: Confirm fine-tuned embedder is available"""
    print("\n" + "="*70)
    print("STEP 3: Verifying Embedder Configuration")
    print("="*70)

    finetuned_path = Path("models/finetuned-embedder")
    if finetuned_path.exists():
        print(f"✓ Fine-tuned model found at: {finetuned_path}")
        print("✓ src/embedder.py will auto-load it on the next import")
        return True

    print("✗ Fine-tuned model directory not found")
    return False


def evaluate_system(hybrid: bool = True):
    """Step 4: Evaluate the entire system with new metrics"""
    print("\n" + "="*70)
    print("STEP 4: Evaluating System")
    print("="*70)
    
    try:
        # Import after embedder is updated
        import importlib
        import sys
        if 'src.embedder' in sys.modules:
            del sys.modules['src.embedder']
        
        from src.evaluation_enhanced import RAGEvaluator
        from src.embedder import embeddings
        
        # Import QA chain (with or without hybrid retrieval)
        if hybrid:
            print("\n✓ Using Hybrid Retrieval (BM25 + Dense + Re-ranking)")
            os.environ["USE_HYBRID_RETRIEVAL"] = "true"
            from src.retrieval_hybrid import qa_chain
        else:
            print("\n✓ Using Dense Retrieval only")
            os.environ["USE_HYBRID_RETRIEVAL"] = "false"
            from src.retrieval import qa_chain
        
        print("\nInitializing evaluator...")
        evaluator = RAGEvaluator(
            qa_chain=qa_chain,
            embeddings=embeddings,
            test_set_path="test_set.json",
        )
        
        print("Running full evaluation (first 20 test cases)...")
        print("-" * 70)
        results = evaluator.run_full_evaluation(limit=20)
        print("-" * 70)
        
        print("\n✓ Evaluation completed!")
        print("  - Results saved to: evaluation_results.json")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_summary():
    """Show summary of changes and improvements"""
    print("\n" + "="*70)
    print("SUMMARY: Improvements Made")
    print("="*70)
    
    improvements = [
        ("✓", "Fine-tuned multilingual-e5-base on legal domain"),
        ("✓", "Added BM25 + Dense hybrid retrieval"),
        ("✓", "Added cross-encoder re-ranking"),
        ("✓", "Enhanced evaluation metrics (NDCG, MRR, Recall, BLEU, ROUGE)"),
        ("✓", "Integrated RAGAS evaluation framework"),
        ("✓", "Created modular training pipeline"),
    ]
    
    for status, improvement in improvements:
        print(f"{status} {improvement}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Compare metrics before/after fine-tuning:
   - Check evaluation_results.json
   - Look for improvements in NDCG, MRR, Recall
   
2. Visualize training results:
   - Check models/finetuned-embedder/training_results.json
   - Plot training loss curves
   
3. Try different configurations:
   - Adjust hybrid_alpha (0-1) for BM25/Dense balance
   - Fine-tune with different batch sizes/epochs
   - Experiment with different loss functions (triplet, multiple_negatives)
   
4. Ablation study:
   - Test without hybrid retrieval: set USE_HYBRID_RETRIEVAL=false
   - Test without re-ranking
   - Test without fine-tuning (using base model)
   
5. Production deployment:
   - Save fine-tuned model
   - Deploy with FastAPI (src/api.py)
   - Update Streamlit UI with new components
    """)


def main():
    """Main training pipeline"""
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Vietnamese Legal RAG - Training Pipeline" + " "*13 + "║")
    print("║" + " "*20 + "Deep Learning Component Setup" + " "*19 + "║")
    print("╚" + "="*68 + "╝")
    
    # Step 0: Install dependencies
    install_dependencies()
    
    # Step 1: Prepare training data
    if not prepare_training_data():
        print("\n✗ Training pipeline failed at Step 1")
        sys.exit(1)
    
    # Step 2: Fine-tune embedding model
    if not finetune_embedding_model():
        print("\n✗ Training pipeline failed at Step 2")
        sys.exit(1)
    
    # Step 3: Update embedder configuration
    if not update_embedder():
        print("\n✗ Training pipeline failed at Step 3")
        sys.exit(1)
    
    # Step 4: Evaluate system
    if not evaluate_system(hybrid=True):
        print("\n⚠ Evaluation had issues but training completed")
    
    # Show summary
    show_summary()
    
    print("\n✓ Training pipeline completed successfully!")
    print("  - Fine-tuned embedding model: models/finetuned-embedder/")
    print("  - Evaluation results: evaluation_results.json")


if __name__ == "__main__":
    main()
