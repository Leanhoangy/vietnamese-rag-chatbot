"""
Augment legal triplets dataset by paraphrasing anchor questions via Groq LLM.

Strategy:
  - 31 unique (anchor, positive) pairs × 3 paraphrases = ~93 new anchors
  - Pair each new anchor with the same positive + existing negatives
  - Result: ~370 triplets (from 93 original)
"""

import json
import os
import random
import re
import time
from pathlib import Path

from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

PARAPHRASE_PROMPT = """Bạn là chuyên gia pháp lý Việt Nam. Hãy viết {n} câu hỏi khác nhau có cùng ý nghĩa với câu hỏi gốc dưới đây. Mỗi câu hỏi phải tự nhiên, đúng ngữ pháp tiếng Việt, và có thể được hỏi bởi người dân thông thường.

Câu hỏi gốc: {question}

Yêu cầu:
- Trả về đúng {n} câu hỏi, mỗi câu trên một dòng
- Không đánh số, không giải thích thêm
- Câu hỏi phải khác nhau về cách diễn đạt nhưng cùng ý nghĩa
- Không thay đổi nghĩa pháp lý của câu hỏi"""


def build_llm(model: str = "llama-3.1-8b-instant") -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in environment")
    return ChatGroq(model=model, api_key=api_key, temperature=0.7, max_retries=2)


def parse_paraphrases(raw: str, n: int) -> list[str]:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    # Remove accidental numbering like "1.", "1)", "-"
    cleaned = [re.sub(r"^[\d]+[.)]\s*|^[-•]\s*", "", l) for l in lines]
    cleaned = [l for l in cleaned if len(l) > 10]
    return cleaned[:n]


def generate_paraphrases(llm: ChatGroq, question: str, n: int = 3, retries: int = 2) -> list[str]:
    prompt = PARAPHRASE_PROMPT.format(question=question, n=n)
    for attempt in range(retries + 1):
        try:
            response = llm.invoke(prompt)
            paraphrases = parse_paraphrases(response.content, n)
            if paraphrases:
                return paraphrases
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(2)
    return []


def augment_dataset(
    dataset_path: str = "legal_triplets_dataset",
    output_path: str = "legal_triplets_dataset",
    paraphrases_per_anchor: int = 3,
    model: str = "llama-3.1-8b-instant",
    delay: float = 0.5,
) -> Dataset:
    print("Loading existing dataset...")
    original_ds = load_from_disk(dataset_path)
    print(f"Original size: {len(original_ds)} triplets")

    # Collect unique (anchor, positive) pairs and their associated negatives
    pair_negatives: dict[tuple[str, str], list[str]] = {}
    for sample in original_ds:
        key = (sample["anchor"], sample["positive"])
        pair_negatives.setdefault(key, [])
        pair_negatives[key].append(sample["negative"])

    unique_pairs = list(pair_negatives.keys())
    print(f"Unique (anchor, positive) pairs: {len(unique_pairs)}")
    print(f"Generating {paraphrases_per_anchor} paraphrases per anchor via Groq ({model})...\n")

    llm = build_llm(model)

    new_anchors: list[str] = []
    new_positives: list[str] = []
    new_negatives: list[str] = []

    for idx, (anchor, positive) in enumerate(unique_pairs):
        print(f"[{idx + 1}/{len(unique_pairs)}] {anchor[:60]}...")
        paraphrases = generate_paraphrases(llm, anchor, n=paraphrases_per_anchor)

        if not paraphrases:
            print("  No paraphrases generated, skipping.")
            time.sleep(delay)
            continue

        negatives = pair_negatives[(anchor, positive)]
        print(f"  Generated {len(paraphrases)} paraphrases, {len(negatives)} negatives available")

        for para in paraphrases:
            for neg in negatives:
                new_anchors.append(para)
                new_positives.append(positive)
                new_negatives.append(neg)

        time.sleep(delay)

    print(f"\nNew augmented triplets: {len(new_anchors)}")

    # Merge original + augmented
    all_anchors = list(original_ds["anchor"]) + new_anchors
    all_positives = list(original_ds["positive"]) + new_positives
    all_negatives = list(original_ds["negative"]) + new_negatives

    # Deduplicate
    seen = set()
    deduped = {"anchor": [], "positive": [], "negative": []}
    for a, p, n in zip(all_anchors, all_positives, all_negatives):
        key = (a, p, n)
        if key not in seen:
            seen.add(key)
            deduped["anchor"].append(a)
            deduped["positive"].append(p)
            deduped["negative"].append(n)

    combined_ds = Dataset.from_dict(deduped)

    # Shuffle
    combined_ds = combined_ds.shuffle(seed=42)

    print(f"Final dataset size: {len(combined_ds)} triplets (after dedup + shuffle)")

    combined_ds.save_to_disk(output_path)
    print(f"Saved to: {output_path}")

    # Summary
    unique_anchors = len(set(combined_ds["anchor"]))
    print(f"\nSummary:")
    print(f"  Original triplets : {len(original_ds)}")
    print(f"  New triplets      : {len(new_anchors)}")
    print(f"  Total (deduped)   : {len(combined_ds)}")
    print(f"  Unique anchors    : {unique_anchors}")

    return combined_ds


if __name__ == "__main__":
    augment_dataset(
        dataset_path="legal_triplets_dataset",
        output_path="legal_triplets_dataset",
        paraphrases_per_anchor=3,
        delay=0.5,
    )
