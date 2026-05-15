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
- k_bm25=10, k_dense=10, k_rerank=5

### 4. Evaluation Framework
- **File:** `src/evaluation_enhanced.py`
- Retrieval: NDCG@5, MRR@10, Recall@5, Precision@5
- Answer quality: BLEU, ROUGE-L, Semantic Similarity
- RAGAS: Faithfulness, Answer Relevancy, Context Precision, Context Recall

---

## Cấu trúc file

```
vietnamese-rag-chatbot/
├── data/
│   └── raw/                          # 5 văn bản luật .docx
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

### Custom Transformer
```bash
python train_custom.py              # data prep → augment → tokenizer → train
rm -rf faiss_index_local/
python -c "from src.vector_store import vectorstore"
python evaluation/evaluate_custom.py
```

### Fine-tuned e5
```bash
python train_finetuned.py           # data prep → fine-tune e5
rm -rf faiss_index_local/
python -c "from src.vector_store import vectorstore"
python evaluation/evaluate_finetuned.py
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
k_bm25 = 10        # BM25 candidates
k_dense = 10       # Dense candidates
k_rerank = 5       # Sau reranking
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
