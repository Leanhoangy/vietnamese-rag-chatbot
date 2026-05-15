# Kiến trúc hệ thống — Vietnamese Legal RAG Chatbot

---

## 1. Tổng quan flow hoạt động

### Offline (chuẩn bị dữ liệu & training)
```
data/raw/*.docx, *.doc, *.pdf
        ↓ loader.py — đọc file
List[Document]
        ↓ chunker.py — chia nhỏ
List[Document] (chunks ~1000 ký tự, overlap 200)
        ↓ embedder.py — chuyển text → vector
List[Vector] (128 chiều hoặc 768 chiều)
        ↓ vector_store.py — lưu vào FAISS
faiss_index_local/  ← index đã build xong
```

### Online (khi user hỏi)
```
Câu hỏi user
        ↓ embedder.py — embed câu hỏi
Vector (128 hoặc 768 chiều)
        ↓ retrieval_hybrid.py
        ├─ BM25 → top-10 chunks (lexical)
        ├─ FAISS → top-10 chunks (semantic)
        ├─ Score fusion (alpha=0.5)
        └─ Cross-Encoder rerank → top-5 chunks
        ↓ llm_chain.py — tạo prompt
Prompt = [System] + [Context: top-5 chunks] + [Question]
        ↓ Groq API (Llama 3.3-70b)
Câu trả lời
```

---

## 2. Các thành phần chính

### 2.1 Loader (`src/loader.py`)
**Tác dụng:** Đọc toàn bộ văn bản luật từ thư mục `data/raw/`

| Class | Xử lý file |
|---|---|
| `Docx2txtLoader` | `.docx` |
| `UnstructuredWordDocumentLoader` | `.doc` |
| `PyPDFLoader` | `.pdf` |

Output: `List[Document]` — mỗi Document chứa `page_content` (text) và `metadata` (source file).

---

### 2.2 Chunker (`src/chunker.py`)
**Tác dụng:** Chia tài liệu dài thành các đoạn nhỏ để embed và retrieve hiệu quả hơn.

**Thuật toán:** `RecursiveCharacterTextSplitter`
- Ưu tiên cắt theo `\n\n` → `\n` → `.` → ` `
- `chunk_size=1000` — tối đa 1000 ký tự/chunk
- `chunk_overlap=200` — 200 ký tự chồng lấp giữa các chunk (tránh mất ngữ cảnh ở biên)

**Lý do overlap:** Thông tin pháp lý thường trải dài qua nhiều câu, overlap giúp không bỏ sót câu quan trọng nằm ở ranh giới 2 chunk.

---

### 2.3 Embedder (`src/embedder.py`)
**Tác dụng:** Chuyển text → vector số để so sánh ngữ nghĩa.

**Logic tự chọn model (ưu tiên từ cao → thấp):**
```
1. Custom Transformer  → models/custom-transformer/model.pt
2. Fine-tuned e5       → models/finetuned-embedder/
3. Base e5             → intfloat/multilingual-e5-base (HuggingFace)
```

**Class `CustomTransformerEmbeddings`:**
- Wrapper LangChain-compatible cho custom transformer
- `embed_documents(texts)` — embed nhiều đoạn văn bản (offline)
- `embed_query(text)` — embed 1 câu hỏi (online)

---

### 2.4 Vector Store (`src/vector_store.py`)
**Tác dụng:** Lưu trữ và tìm kiếm vector bằng FAISS.

**FAISS (Facebook AI Similarity Search):**
- Lưu toàn bộ chunk vectors vào index
- Tìm kiếm bằng **cosine similarity** / **L2 distance**
- Độ phức tạp: O(n) brute force, có thể dùng IVF/HNSW cho dataset lớn
- Lưu local vào `faiss_index_local/` để tái sử dụng

```python
# Lần đầu: build từ documents
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local(INDEX_PATH)

# Lần sau: load lại
vectorstore = FAISS.load_local(INDEX_PATH, embeddings)
```

---

### 2.5 Hybrid Retriever (`src/retrieval_hybrid.py`)

#### BM25 (Sparse Retrieval)
**Thuật toán:** Best Match 25 — tìm kiếm từ khóa có trọng số TF-IDF cải tiến
```
Score(Q, D) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1×(1-b+b×|D|/avgdl))
```
- `f(qi, D)` = tần suất từ qi trong document D
- `k1=1.5`, `b=0.75` — tham số điều chỉnh
- **Ưu điểm:** Tốt với từ khóa chính xác (số điều, mức phạt cụ thể)
- **Nhược điểm:** Không hiểu ngữ nghĩa ("phạt tiền" vs "nộp phạt")

#### Dense Retrieval (FAISS)
- Tìm kiếm dựa trên vector embedding
- **Ưu điểm:** Hiểu ngữ nghĩa, đồng nghĩa
- **Nhược điểm:** Có thể bỏ sót từ khóa chính xác

#### Score Fusion
```python
fused_score = alpha × dense_score + (1 - alpha) × bm25_score
# alpha = 0.5 → cân bằng 50-50
```

#### Cross-Encoder Reranking
**Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual)

**Khác biệt với Bi-Encoder (FAISS):**
```
Bi-Encoder:  embed(query) ←cosine→ embed(document)   # nhanh, embed riêng lẻ
Cross-Encoder: score(query + document)                 # chậm hơn, chính xác hơn
```
Cross-Encoder đọc cả query lẫn document cùng lúc → hiểu mối quan hệ sâu hơn → rerank chính xác hơn.

---

### 2.6 LLM Chain (`src/llm_chain.py`)
**Tác dụng:** Kết nối Groq API để sinh câu trả lời.

**Model:** `llama-3.3-70b-versatile`
- `temperature=0` — output deterministic, không sáng tạo tùy tiện
- `max_tokens=1024` — giới hạn độ dài câu trả lời

**Prompt template (trong `retrieval_hybrid.py`):**
```
NGUYÊN TẮC BẮT BUỘC:
- Chỉ sử dụng thông tin có trong TÀI LIỆU
- Nếu không đủ thông tin → nói rõ "Tài liệu không cung cấp đủ thông tin"
- Trích dẫn chính xác số liệu, điều khoản
- Không suy luận ngoài phạm vi tài liệu
```

---

## 3. Custom Transformer (`src/custom_transformer/`)

### 3.1 Tokenizer (`tokenizer.py`)
**Tác dụng:** Chuyển text → dãy số nguyên

**Quy trình build:**
```
text → lowercase → xóa ký tự đặc biệt → split theo space
→ đếm tần suất → lấy 8000 từ phổ biến nhất → lưu vocab.json
```

**Special tokens:**
| Token | ID | Tác dụng |
|---|---|---|
| `[PAD]` | 0 | Padding đủ độ dài |
| `[UNK]` | 1 | Từ không có trong vocab |
| `[CLS]` | 2 | Bắt đầu câu |
| `[SEP]` | 3 | Kết thúc câu |

**Encode:**
```
"Vượt đèn đỏ bị phạt"
→ ["[CLS]", "vượt", "đèn", "đỏ", "bị", "phạt", "[SEP]"]
→ [2, 145, 302, 89, 67, 4, 3, 0, 0, ..., 0]  (padding đủ 128)
```

---

### 3.2 Kiến trúc Transformer (`architecture.py`)

```
Input: input_ids (batch, 128)  +  attention_mask (batch, 128)
         ↓
Token Embedding         → (batch, 128, 128)   # map token_id → vector 128 chiều
         ↓
Positional Encoding     → (batch, 128, 128)   # thêm thông tin vị trí
         ↓
TransformerEncoderLayer × 2
  ├─ Pre-LayerNorm
  ├─ Multi-Head Self-Attention (4 heads)
  ├─ Residual Connection
  ├─ LayerNorm
  ├─ Feed-Forward (128 → 512 → 128)
  └─ Residual Connection
         ↓
Mean Pooling            → (batch, 128)         # trung bình các token (bỏ PAD)
         ↓
L2 Normalize            → (batch, 128)         # đưa về unit vector
         ↓
Output: sentence vector 128 chiều
```

**Positional Encoding (sin/cos):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
Giúp model biết vị trí của từng token trong câu.

**Multi-Head Self-Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
- 4 heads → học 4 pattern quan hệ khác nhau giữa các từ
- `√d_k` = scale để tránh gradient vanishing

---

### 3.3 Trainer (`trainer.py`)

**Triplet Loss:**
```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```
- `d(x, y)` = khoảng cách Euclidean giữa 2 vector
- `margin = 0.3` — khoảng cách tối thiểu giữa positive và negative
- Loss = 0 khi model đã phân biệt đúng với margin đủ lớn

**Optimizer:** AdamW
```
θ = θ - lr × (gradient + weight_decay × θ)
```
- `lr = 3e-4`
- `weight_decay = 1e-2` — L2 regularization tránh overfit

**Scheduler:** CosineAnnealingLR — giảm learning rate theo hàm cosine qua các epoch.

---

## 4. Data Pipeline

### 4.1 Data Preparation (`src/data_preparation.py`)

**Class `LegalDataPreparator`:**

| Hàm | Tác dụng |
|---|---|
| `load_test_cases()` | Đọc 68 câu hỏi từ `test_set.json` |
| `prepare_positive_pairs()` | Tạo cặp (câu hỏi, đáp án đúng) |
| `mine_hard_negatives()` | Tìm đoạn văn SAI nhưng nghe có vẻ đúng |
| `prepare_triplets()` | Ghép thành (anchor, positive, negative) |
| `create_dataset()` | Xuất HuggingFace Dataset |

**Hard Negative Mining:**
```
query = "Vượt đèn đỏ phạt bao nhiêu?"
→ FAISS.similarity_search(query, k=10)
→ lọc bỏ ground truth
→ còn lại = hard negatives (liên quan nhưng không phải đáp án)
```
Hard negatives khó hơn random negatives → model học tốt hơn.

---

### 4.2 Augment Triplets (`src/augment_triplets.py`)

**Tác dụng:** Tăng dataset từ ~204 → ~807 triplets bằng cách paraphrase câu hỏi qua LLM.

```
"Vượt đèn đỏ bị phạt bao nhiêu?"
→ Groq LLM paraphrase × 3
→ "Mức phạt khi không chấp hành đèn tín hiệu là gì?"
→ "Đi qua đèn đỏ sẽ bị xử lý như thế nào?"
→ "Vi phạm đèn giao thông bị phạt tiền bao nhiêu?"
```
Mỗi paraphrase + positive + negatives gốc = triplets mới.

---

## 5. Evaluation (`src/evaluation_enhanced.py`)

### Class `RAGEvaluator`

| Hàm | Tác dụng |
|---|---|
| `load_test_set()` | Đọc test_set.json |
| `run_qa_pipeline()` | Chạy toàn bộ RAG chain trên test set |
| `evaluate_retrieval()` | Tính NDCG, MRR, Recall, Precision |
| `evaluate_ragas()` | Tính Faithfulness, Answer Relevancy, Context Precision/Recall |
| `evaluate_answer_quality()` | Tính BLEU, ROUGE-L, Semantic Similarity |
| `run_full_evaluation()` | Chạy tất cả, lưu `evaluation_results.json` |

**NDCG@5 (Normalized Discounted Cumulative Gain):**
```
DCG@k = Σ rel_i / log2(i+1)      # i = vị trí trong top-k
NDCG@k = DCG@k / IDCG@k          # chuẩn hóa về [0,1]
```

**MRR@10 (Mean Reciprocal Rank):**
```
MRR = (1/|Q|) × Σ 1/rank_i       # rank_i = vị trí document đúng đầu tiên
```

---

## 6. API & UI

### FastAPI (`src/api.py`)
- `POST /chat` — nhận câu hỏi, trả về câu trả lời + source documents
- `GET /health` — kiểm tra server

### Streamlit (`src/app.py`)
- Giao diện chat web
- Hiển thị câu trả lời + tài liệu nguồn

---

Đồ án môn Deep Learning — HUIT 2025
