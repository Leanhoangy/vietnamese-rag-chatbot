# Vietnamese RAG Chatbot — Tư Vấn Pháp Luật

> Chatbot hỏi đáp pháp luật Việt Nam sử dụng Retrieval-Augmented Generation (RAG)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://langchain.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vietnamese-rag-chatbot-jemrcpipbmxd5bdugx5z2l.streamlit.app)

## Live Demo

👉 [vietnamese-rag-chatbot-jemrcpipbmxd5bdugx5z2l.streamlit.app](https://vietnamese-rag-chatbot-jemrcpipbmxd5bdugx5z2l.streamlit.app)

---

## Giới thiệu

Hệ thống chatbot cho phép người dùng hỏi đáp về pháp luật Việt Nam dựa trên các văn bản luật chính thức (Luật Đường bộ 2024, Bộ luật Lao động, Luật Doanh nghiệp, Luật Dân sự, Nghị định 100/2019).

---

## Kiến trúc RAG Pipeline

```
Tài liệu .docx
      ↓
[Chunking] RecursiveCharacterTextSplitter (chunk_size=1000, overlap=300)
      ↓
[Embedding] Custom Transformer (tự xây) hoặc Fine-tuned multilingual-e5-base
      ↓
[Vector Store] FAISS

── Khi user hỏi ──

Câu hỏi → BM25 + FAISS dense retrieval → Cross-Encoder re-rank (multilingual)
      ↓
Prompt → Groq (Llama 3.3-70b) → Câu trả lời
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain 0.3 |
| Embedding | Custom Transformer (PyTorch) / Fine-tuned multilingual-e5-base |
| Vector Store | FAISS |
| Retrieval | Hybrid BM25 + Dense + Cross-Encoder reranking |
| LLM | Llama 3.3-70b (Groq API) |
| UI | Streamlit |
| API | FastAPI |
| Evaluation | RAGAS + NDCG + MRR + BLEU + ROUGE |

---

## Hai hướng Embedding

### 1. Custom Transformer (tự xây từ đầu)
- Kiến trúc: Transformer Encoder 2 layers, 4 heads, hidden_dim=128
- Tokenizer: xây từ tài liệu pháp lý (~8000 từ)
- Train: Triplet Loss trên legal triplets dataset
- Tham số: ~711,000

### 2. Fine-tuned multilingual-e5-base (transfer learning)
- Base model: `intfloat/multilingual-e5-base` (278M params)
- Fine-tune: trên legal triplets với hard negative mining
- Data augmentation: paraphrase câu hỏi qua Groq LLM (~807 triplets)

---

## Cài đặt và chạy

```bash
# Clone repo
git clone https://github.com/Leanhoangy/vietnamese-rag-chatbot.git
cd vietnamese-rag-chatbot

# Cài thư viện
pip install -r requirements.txt

# Tạo file .env
echo "GROQ_API_KEY=your_key_here" > .env
```

### Train embedding model

```bash
# Custom transformer (tự xây)
python train_custom.py

# Fine-tune multilingual-e5-base
python train_finetuned.py
```

### Build FAISS index

```bash
python -c "from src.vector_store import vectorstore; print('Done')"
```

### Chạy app

```bash
# Streamlit UI
streamlit run src/app.py

# FastAPI (tùy chọn — nếu muốn dùng REST API độc lập)
uvicorn src.api:app --reload
```

### Evaluation

```bash
# Đánh giá custom transformer
python evaluation/evaluate_custom.py

# Đánh giá fine-tuned e5
python evaluation/evaluate_finetuned.py
```

---

## Cấu trúc project

```
vietnamese-rag-chatbot/
├── data/
│   └── raw/                        # Văn bản luật .docx
├── src/
│   ├── custom_transformer/         # Model tự xây từ đầu
│   │   ├── tokenizer.py            # Xây vocab, encode text
│   │   ├── architecture.py         # Kiến trúc transformer
│   │   └── trainer.py              # Train với Triplet Loss
│   ├── finetuned_e5/               # Fine-tune từ pretrained model
│   │   └── trainer.py              # Fine-tune multilingual-e5-base
│   ├── loader.py                   # Đọc file .docx
│   ├── chunker.py                  # Chia text thành chunks
│   ├── embedder.py                 # Load embedding model (auto-select)
│   ├── vector_store.py             # Build/load FAISS index
│   ├── llm_chain.py                # Kết nối Groq API
│   ├── retrieval_hybrid.py         # Hybrid BM25 + Dense + Cross-Encoder
│   ├── data_preparation.py         # Tạo triplets training
│   ├── augment_triplets.py         # Augment data qua LLM
│   ├── evaluation_enhanced.py      # RAGAS + Retrieval metrics
│   ├── app.py                      # Streamlit UI
│   └── api.py                      # FastAPI endpoint
├── evaluation/
│   ├── evaluate_custom.py          # Đánh giá custom transformer
│   └── evaluate_finetuned.py       # Đánh giá fine-tuned e5
├── train_custom.py                 # Pipeline train custom transformer
├── train_finetuned.py              # Pipeline fine-tune e5
├── test_set.json                   # 68 câu hỏi test
└── requirements.txt
```

---

## Kết quả Evaluation

So sánh 2 embedding model với Hybrid Retrieval (BM25 + FAISS + Cross-Encoder, k=7):

### Retrieval Metrics
| Metric           | Custom Transformer | Fine-tuned e5 |
|:-----------------|:------------------:|:-------------:|
| NDCG@7           | 0.9937             | **0.9950**    |
| MRR@10           | 1.0000             | **1.0000**    |
| Recall@7         | 1.0000             | **1.0000**    |
| Precision@7      | 0.9821             | **0.9857**    |

### RAGAS Metrics
| Metric            | Custom Transformer | Fine-tuned e5 |
|:------------------|:------------------:|:-------------:|
| Faithfulness      | 0.7738             | **0.8854**    |
| Answer Relevancy  | 0.8654             | **0.9104**    |
| Context Precision | 0.6160             | **0.6820**    |
| Context Recall    | 0.7500             | **0.9000**    |

### Answer Quality
| Metric             | Custom Transformer | Fine-tuned e5 |
|:-------------------|:------------------:|:-------------:|
| BLEU               | 0.1423             | **0.1684**    |
| ROUGE-L            | 0.3539             | **0.3682**    |
| Semantic Similarity| 0.7428             | **0.8049**    |

> Fine-tuned e5 (278M params) vượt trội Custom Transformer (711K params) trên toàn bộ metrics.

---

## Nhóm thực hiện

Đồ án môn Deep Learning — HUIT 2025
