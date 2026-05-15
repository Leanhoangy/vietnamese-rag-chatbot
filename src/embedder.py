import json
from pathlib import Path

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

CUSTOM_MODEL_PATH = Path("models/custom-transformer/model.pt")
CUSTOM_CONFIG_PATH = Path("models/custom-transformer/config.json")
CUSTOM_TOKENIZER_PATH = Path("models/custom-tokenizer/vocab.json")

FINETUNED_MODEL_PATH = Path("models/finetuned-embedder")
FINETUNED_CONFIG_PATH = FINETUNED_MODEL_PATH / "config.json"


class CustomTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper cho custom transformer."""

    def __init__(self, model, tokenizer, max_len: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _encode_text(self, text: str) -> list[float]:
        enc = self.tokenizer.encode_plus(text, self.max_len)
        input_ids = torch.tensor([enc["input_ids"]])
        attention_mask = torch.tensor([enc["attention_mask"]])
        vec = self.model.encode(input_ids, attention_mask)
        return vec[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode_text(text)


def _load_custom_transformer() -> CustomTransformerEmbeddings | None:
    if not (CUSTOM_MODEL_PATH.exists() and CUSTOM_CONFIG_PATH.exists() and CUSTOM_TOKENIZER_PATH.exists()):
        return None
    from src.custom_transformer.tokenizer import LegalTokenizer
    from src.custom_transformer.architecture import LegalTransformerEncoder

    with open(CUSTOM_CONFIG_PATH) as f:
        cfg = json.load(f)

    tokenizer = LegalTokenizer()
    tokenizer.load(str(CUSTOM_TOKENIZER_PATH))

    model = LegalTransformerEncoder(
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        max_len=cfg["max_len"],
    )
    state = torch.load(str(CUSTOM_MODEL_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return CustomTransformerEmbeddings(model, tokenizer, cfg["max_len"])


def _has_valid_finetuned_model() -> bool:
    if not FINETUNED_MODEL_PATH.exists() or not FINETUNED_CONFIG_PATH.exists():
        return False
    try:
        with open(FINETUNED_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return "model_type" in config


# Ưu tiên: custom transformer → fine-tuned e5 → base e5
custom = _load_custom_transformer()
if custom is not None:
    print("Loading custom transformer embedding model...")
    embeddings = custom
elif _has_valid_finetuned_model():
    print("Loading fine-tuned embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=str(FINETUNED_MODEL_PATH),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
else:
    print("Using base embedding model (multilingual-e5-base).")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
