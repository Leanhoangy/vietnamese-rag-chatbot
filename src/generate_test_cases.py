"""
Generate additional test cases from raw legal documents using Groq LLM.

Strategy:
  - Parse each .docx into article-level chunks (Điều X...)
  - Sample articles (weighted toward under-represented domains)
  - Ask LLM to produce 1-2 Q&A pairs per article
  - ground_truth must be an exact substring of the article text
  - Merge with existing test_set.json and deduplicate
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

from docx import Document
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
TEST_SET_PATH = Path("test_set.json")

DOC_META = {
    "Luatdansu.docx":       {"label": "Bộ luật Dân sự",    "domain": "Dân sự"},
    "Luatdoanhnghiep.docx": {"label": "Luật Doanh nghiệp", "domain": "Doanh nghiệp"},
    "Luatlaodong.docx":     {"label": "Bộ luật Lao động",  "domain": "Lao động"},
    "Luatduongbo2024.docx": {"label": "Luật Đường bộ",     "domain": "Giao thông"},
    "Nghidinh 100:2019.docx": {"label": "Nghị định 100/2019", "domain": "Giao thông"},
}

# How many extra questions to target per domain
DOMAIN_TARGETS = {
    "Dân sự":      12,
    "Doanh nghiệp": 9,
    "Lao động":     6,
    "Giao thông":   5,
}

QA_PROMPT = """Bạn là chuyên gia pháp lý Việt Nam. Đọc đoạn văn bản pháp luật sau và tạo ra {n} cặp hỏi-đáp mà người dân thông thường có thể hỏi.

Văn bản:
\"\"\"
{text}
\"\"\"

Yêu cầu NGHIÊM NGẶT:
1. Câu hỏi phải tự nhiên, như người dân hỏi (không dùng thuật ngữ "điều X", "khoản Y")
2. Câu trả lời phải là đoạn trích NGUYÊN VĂN từ văn bản trên (copy chính xác)
3. Câu trả lời phải ngắn gọn, đủ thông tin (1-3 câu)
4. Trả về JSON array, mỗi phần tử có 2 key: "question" và "ground_truth"
5. Chỉ trả về JSON, không giải thích thêm

Ví dụ format:
[
  {{"question": "Câu hỏi 1?", "ground_truth": "Trích dẫn nguyên văn từ văn bản"}},
  {{"question": "Câu hỏi 2?", "ground_truth": "Trích dẫn nguyên văn khác"}}
]"""


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

def extract_articles(docx_path: Path) -> list[dict]:
    """Split a .docx into article-level chunks keyed by 'Điều X'."""
    doc = Document(docx_path)
    articles = []
    current_article: Optional[str] = None
    current_lines: list[str] = []

    article_re = re.compile(r"^(Điều\s+\d+[\.\:]?)", re.IGNORECASE)

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        m = article_re.match(text)
        if m:
            if current_article and current_lines:
                articles.append({
                    "title": current_article,
                    "text": "\n".join(current_lines),
                })
            current_article = text
            current_lines = [text]
        elif current_article:
            current_lines.append(text)

    if current_article and current_lines:
        articles.append({
            "title": current_article,
            "text": "\n".join(current_lines),
        })

    return articles


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def build_llm(model: str = "llama-3.1-8b-instant") -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in environment")
    return ChatGroq(model=model, api_key=api_key, temperature=0.3, max_retries=2)


def parse_qa_json(raw: str) -> list[dict]:
    """Extract JSON array from LLM response, tolerating leading text and markdown fences."""
    # Find the first '[' and last ']' to extract the JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
        if isinstance(data, list):
            return [
                item for item in data
                if isinstance(item, dict)
                and "question" in item
                and "ground_truth" in item
                and len(item["question"]) > 10
                and len(item["ground_truth"]) > 15
            ]
    except json.JSONDecodeError:
        pass
    return []


def ground_truth_in_text(ground_truth: str, article_text: str) -> bool:
    """Check that key terms from ground_truth appear in source text (grounding check)."""
    text_clean = re.sub(r"\s+", " ", article_text).lower()
    gt_clean = re.sub(r"\s+", " ", ground_truth).lower()

    # Extract meaningful tokens (≥4 chars) from ground_truth
    tokens = [t for t in re.findall(r"\w+", gt_clean) if len(t) >= 4]
    if not tokens:
        return False

    # At least 60% of key tokens must appear in the article
    matched = sum(1 for t in tokens if t in text_clean)
    return matched / len(tokens) >= 0.6


def generate_qa_pairs(
    llm: ChatGroq,
    article: dict,
    n: int = 2,
    retries: int = 2,
) -> list[dict]:
    # Skip very short articles (less than 100 chars — likely just a title)
    if len(article["text"]) < 100:
        return []

    prompt = QA_PROMPT.format(text=article["text"][:2000], n=n)

    for attempt in range(retries + 1):
        try:
            response = llm.invoke(prompt)
            pairs = parse_qa_json(response.content)
            # Validate ground_truth is actually in the article
            valid = [
                p for p in pairs
                if ground_truth_in_text(p["ground_truth"], article["text"])
            ]
            if valid:
                return valid
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(2)

    return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_existing_test_set(path: Path) -> list[dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def existing_questions(test_set: list[dict]) -> set[str]:
    return {item["question"].strip().lower() for item in test_set}


def generate_test_cases(
    delay: float = 0.8,
    articles_per_domain_batch: int = 6,
    model: str = "llama-3.1-8b-instant",
) -> list[dict]:
    existing = load_existing_test_set(TEST_SET_PATH)
    seen_questions = existing_questions(existing)

    domain_counts: dict[str, int] = {}
    for item in existing:
        source = item.get("source", "")
        for domain, target in DOMAIN_TARGETS.items():
            meta_labels = [v["label"] for k, v in DOC_META.items() if v["domain"] == domain]
            if any(lbl in source for lbl in meta_labels) or domain == "Giao thông" and any(
                kw in source for kw in ["Đường bộ", "Nghị định", "168", "100"]
            ):
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                break

    print("Current domain distribution:")
    for domain, target in DOMAIN_TARGETS.items():
        current = domain_counts.get(domain, 0)
        print(f"  {domain}: {current} (target +{target} more)")

    llm = build_llm(model)
    new_cases: list[dict] = []

    for filename, meta in DOC_META.items():
        domain = meta["domain"]
        label = meta["label"]
        target_new = DOMAIN_TARGETS.get(domain, 0)

        if target_new <= 0:
            continue

        docx_path = RAW_DIR / filename
        if not docx_path.exists():
            print(f"File not found: {docx_path}")
            continue

        print(f"\n=== {label} ({domain}) — target +{target_new} ===")
        articles = extract_articles(docx_path)
        print(f"  Extracted {len(articles)} articles")

        # Filter articles with substantive content (>200 chars, contains numbers or rules)
        good_articles = [
            a for a in articles
            if len(a["text"]) > 200
            and re.search(r"\d", a["text"])
        ]
        print(f"  Substantive articles: {len(good_articles)}")

        # Sample articles — slightly more than needed to account for failures
        sample_size = min(len(good_articles), target_new * articles_per_domain_batch)
        sampled = random.sample(good_articles, sample_size)

        domain_new = 0
        for i, article in enumerate(sampled):
            if domain_new >= target_new:
                break

            remaining = target_new - domain_new
            n_pairs = min(2, remaining)

            print(f"  [{i + 1}/{len(sampled)}] {article['title'][:60]}...")
            pairs = generate_qa_pairs(llm, article, n=n_pairs)

            for pair in pairs:
                q = pair["question"].strip()
                if q.lower() in seen_questions:
                    print(f"    Duplicate skipped: {q[:50]}")
                    continue

                seen_questions.add(q.lower())
                new_cases.append({
                    "question": q,
                    "ground_truth": pair["ground_truth"].strip(),
                    "source": label + " - " + article["title"],
                })
                domain_new += 1
                print(f"    + Q: {q[:70]}")

                if domain_new >= target_new:
                    break

            time.sleep(delay)

        print(f"  Generated {domain_new} new cases for {domain}")

    return new_cases


def main():
    random.seed(42)

    print("Loading existing test set...")
    existing = load_existing_test_set(TEST_SET_PATH)
    print(f"Existing: {len(existing)} test cases\n")

    new_cases = generate_test_cases(delay=0.8)

    combined = existing + new_cases
    combined_unique = list({item["question"]: item for item in combined}.values())

    print(f"\n=== Summary ===")
    print(f"  Original : {len(existing)}")
    print(f"  New      : {len(new_cases)}")
    print(f"  Total    : {len(combined_unique)}")

    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_unique, f, ensure_ascii=False, indent=2)

    print(f"Saved to {TEST_SET_PATH}")


if __name__ == "__main__":
    main()
