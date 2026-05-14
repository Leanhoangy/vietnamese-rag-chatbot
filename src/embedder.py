import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings


FINETUNED_MODEL_PATH = Path("models/finetuned-embedder")
FINETUNED_CONFIG_PATH = FINETUNED_MODEL_PATH / "config.json"


def _has_valid_finetuned_model() -> bool:
    """Load the fine-tuned model only when the directory contains a real model."""
    if not FINETUNED_MODEL_PATH.exists() or not FINETUNED_CONFIG_PATH.exists():
        return False

    try:
        with open(FINETUNED_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return "model_type" in config

if _has_valid_finetuned_model():
    print("Loading fine-tuned embedding model...")
    model_name = str(FINETUNED_MODEL_PATH)
else:
    print("Fine-tuned model not found. Using base embedding model.")
    model_name = "intfloat/multilingual-e5-base"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
