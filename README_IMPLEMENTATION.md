# Chi tiết triển khai — Vietnamese Legal RAG Chatbot

---

## Các thành phần Deep Learning

### 1. Custom Transformer Encoder (tự xây từ đầu)
- **File:** `src/custom_transformer/`
- Kiến trúc: Token Embedding + Positional Encoding → TransformerEncoderLayer × 2 → Mean Pooling → L2 Normalize
- Config: hidden_dim=128, num_heads=4, ff_dim=512, max_len=128
- Tokenizer: xây vocabulary ~8000 từ từ tài liệu pháp lý
- Loss: TripletLoss(margin=0.3)
- Optimizer: AdamW + CosineAnnealingLR
- Tham số: ~711,000

### 2. Fine-tuned multilingual-e5-base (transfer learning)
- **File:** `src/finetuned_e5/`
- Base model: `intfloat/multilingual-e5-base` (278M params)
- Chiến lược: Hard negative mining từ FAISS index
- Data augmentation: paraphrase câu hỏi qua Groq LLM → 807 triplets
- Loss: TripletLoss

### 3. Hybrid Retrieval System
- **File:** `src/retrieval_hybrid.py`
- BM25 (sparse) + FAISS dense retrieval → score fusion (alpha=0.5)
- Cross-Encoder reranking: `mmarco-mMiniLMv2-L12-H384-v1` (multilingual)
- k=7, k_bm25=14, k_dense=14, k_rerank=7

### 4. Evaluation Framework
- **File:** `src/evaluation_enhanced.py`
- Retrieval: NDCG@7, MRR@10, Recall@7, Precision@7
- Answer quality: BLEU, ROUGE-L, Semantic Similarity
- RAGAS: Faithfulness, Answer Relevancy, Context Precision, Context Recall

---

## Cấu trúc file

```
vietnamese-rag-chatbot/
├── data/
│   └── raw/                          # 8 văn bản luật .docx
├── src/
│   ├── custom_transformer/           # Model tự xây
│   │   ├── __init__.py
│   │   ├── tokenizer.py              # Xây vocab, encode text → số
│   │   ├── architecture.py           # Kiến trúc transformer (PyTorch)
│   │   └── trainer.py                # Train với Triplet Loss
│   ├── finetuned_e5/                 # Fine-tune từ pretrained
│   │   ├── __init__.py
│   │   └── trainer.py                # Fine-tune multilingual-e5-base
│   ├── loader.py                     # Đọc file .docx
│   ├── chunker.py                    # Chia text thành chunks (1000 ký tự)
│   ├── embedder.py                   # Auto-select embedding model
│   ├── vector_store.py               # Build/load FAISS index
│   ├── llm_chain.py                  # Groq API (Llama 3.3-70b)
│   ├── retrieval_hybrid.py           # Hybrid BM25 + Dense + Cross-Encoder
│   ├── data_preparation.py           # Tạo triplets từ test set
│   ├── augment_triplets.py           # Augment data qua LLM paraphrase
│   ├── generate_test_cases.py        # Tạo test_set.json từ văn bản luật
│   ├── evaluation_enhanced.py        # RAGAS + Retrieval metrics
│   ├── app.py                        # Streamlit UI
│   └── api.py                        # FastAPI endpoint
├── evaluation/
│   ├── evaluate_custom.py            # Đánh giá custom transformer
│   └── evaluate_finetuned.py         # Đánh giá fine-tuned e5
├── train_custom.py                   # Pipeline train custom transformer
├── train_finetuned.py                # Pipeline fine-tune e5
├── test_set.json                     # 68 câu hỏi test
└── requirements.txt
```

---

## Thứ tự chạy

### Bước 1 — Tạo test set (chỉ cần chạy 1 lần)
```bash
python src/generate_test_cases.py
# Đọc data/raw/*.docx → hỏi Groq LLM tạo Q&A → lưu test_set.json (68 câu hỏi)
```

### Bước 2 — Build FAISS index lần đầu (dùng base e5)
```bash
python -c "from src.vector_store import vectorstore"
# loader.py đọc .docx → chunker.py chia chunks → embedder.py embed → lưu faiss_index_local/
```

### Bước 3 — Chuẩn bị training data
```bash
# Được gọi tự động bên trong train_custom.py / train_finetuned.py, bao gồm:
# data_preparation.py  → đọc test_set.json + dùng FAISS mine hard negatives → legal_triplets_dataset/
# augment_triplets.py  → paraphrase câu hỏi qua LLM → 68 → 807 triplets
```

### Bước 4 — Train model (chọn 1 trong 2)
```bash
python train_custom.py      # tự động: data prep → augment → build tokenizer → train (711K params)
# HOẶC
python train_finetuned.py   # tự động: data prep → augment → fine-tune e5 (278M params)
```

### Bước 5 — Rebuild FAISS với model mới
```bash
rm -rf faiss_index_local/
python -c "from src.vector_store import vectorstore"
# embedder.py tự chọn: fine-tuned e5 > custom transformer > base e5
```

### Bước 6 — Đánh giá
```bash
python evaluation/evaluate_custom.py      # custom transformer: retrieval + RAGAS metrics
python evaluation/evaluate_finetuned.py   # fine-tuned e5: retrieval + RAGAS metrics
```

### Bước 7 — Chạy chatbot
```bash
streamlit run src/app.py
```

---

## Cấu hình quan trọng

### Custom Transformer (`src/custom_transformer/trainer.py`)
```python
CONFIG = {
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "ff_dim": 512,
    "epochs": 5,
    "batch_size": 16,
    "lr": 3e-4,
    "triplet_margin": 0.3,
}
```

### Hybrid Retrieval (`src/retrieval_hybrid.py`)
```python
k = 7              # Final top-k results
k_bm25 = k * 2    # BM25 candidates (14)
k_dense = k * 2   # Dense candidates (14)
k_rerank = k      # Sau reranking (7)
alpha = 0.5        # 0=pure BM25, 1=pure dense
```

### LLM (`src/llm_chain.py`)
```python
model = "llama-3.3-70b-versatile"
temperature = 0
max_tokens = 1024
```

---

## Logic tự chọn embedding (`src/embedder.py`)

```
Ưu tiên:
1. Custom Transformer  (nếu models/custom-transformer/model.pt tồn tại)
2. Fine-tuned e5       (nếu models/finetuned-embedder/ tồn tại)
3. Base e5             (fallback — intfloat/multilingual-e5-base)
```

---

## Giá trị học thuật

| Yêu cầu đồ án | Đáp ứng |
|---|---|
| Có thành phần Deep Learning | Transformer tự xây + TripletLoss + backpropagation |
| Có NLP/Transformer | Custom encoder + Cross-Encoder reranking |
| Có pipeline training | train_custom.py + train_finetuned.py |
| Có evaluation | 10+ metrics: NDCG, MRR, BLEU, ROUGE, RAGAS |
| Không chỉ gọi API | Tự xây model, tự train, tự evaluate |
| Có thể so sánh | Ablation: base e5 vs fine-tuned e5 vs custom transformer |

---

## Chuẩn bị thuyết trình

1. **Data pipeline** — test_set.json → hard negative mining → 807 triplets
2. **Custom Transformer** — giải thích kiến trúc, TripletLoss, mean pooling
3. **Fine-tuning** — transfer learning, tại sao e5-base là base tốt
4. **Hybrid Retrieval** — tại sao BM25 + Dense tốt hơn chỉ 1 trong 2
5. **So sánh 3 hướng** — bảng metric: base vs fine-tuned vs custom
6. **Ablation study** — đóng góp của từng thành phần

---

Đồ án môn Deep Learning — HUIT 2025
