"""
Tokenizer đơn giản cho văn bản pháp lý tiếng Việt.
Xây vocabulary từ tài liệu raw, tokenize theo từ.
"""

import json
import re
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str):
    return _normalize(text).split()


class LegalTokenizer:
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_texts(self, texts: list[str]) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(_tokenize(text))

        most_common = [tok for tok, _ in counter.most_common(self.vocab_size - len(SPECIAL_TOKENS))]
        vocab = SPECIAL_TOKENS + most_common

        self.token2id = {tok: idx for idx, tok in enumerate(vocab)}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}
        print(f"Vocabulary size: {len(self.token2id)}")

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str, max_length: int = 128) -> list[int]:
        tokens = _tokenize(text)
        ids = [CLS_ID]
        for tok in tokens:
            ids.append(self.token2id.get(tok, UNK_ID))
        ids.append(SEP_ID)

        # truncate (keep [CLS] and [SEP])
        if len(ids) > max_length:
            ids = ids[: max_length - 1] + [SEP_ID]

        return ids

    def pad(self, ids: list[int], max_length: int = 128) -> tuple[list[int], list[int]]:
        attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
        ids = ids + [PAD_ID] * (max_length - len(ids))
        return ids, attention_mask

    def encode_plus(self, text: str, max_length: int = 128) -> dict:
        ids = self.encode(text, max_length)
        ids, mask = self.pad(ids, max_length)
        return {"input_ids": ids, "attention_mask": mask}

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str = "models/custom-tokenizer/vocab.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved → {path}")

    def load(self, path: str = "models/custom-tokenizer/vocab.json") -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        print(f"Tokenizer loaded — vocab size: {self.vocab_size}")

    def __len__(self) -> int:
        return len(self.token2id)


# ------------------------------------------------------------------
# Build tokenizer từ legal documents khi chạy trực tiếp
# ------------------------------------------------------------------

def build_tokenizer() -> LegalTokenizer:
    from src.chunker import splits

    texts = [doc.page_content for doc in splits]
    print(f"Building tokenizer from {len(texts)} chunks...")

    tokenizer = LegalTokenizer(vocab_size=8000)
    tokenizer.build_from_texts(texts)
    tokenizer.save()
    return tokenizer


if __name__ == "__main__":
    build_tokenizer()
