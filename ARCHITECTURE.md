# Kiến trúc hệ thống — Vietnamese Legal RAG Chatbot

Tài liệu này mô tả kiến trúc **hiện tại** của hệ thống (cập nhật 10/2026). Phần tổng quan bằng tiếng Anh nằm ở [README.md](README.md).

---

## 1. Tổng quan

### Offline — chuẩn bị dữ liệu
```
data/raw/*.docx (8 văn bản luật)
        ↓ loader.py      đọc file (.docx; .doc/.pdf nếu có)
        ↓ chunker.py     tách theo Điều → gom theo khoản, gắn nhãn "[Tên văn bản] Điều N."
        ↓ embedder.py    fine-tuned multilingual-e5-base (768 chiều) hoặc Custom Transformer (128 chiều)
        ↓ vector_store.py
faiss_index_local/        index FAISS đã build (768 chiều, fine-tuned E5) — commit kèm repo cho Streamlit Cloud
```

### Online — trả lời một câu hỏi
```
Câu hỏi (+ 5 lượt hội thoại gần nhất)
        ↓ legal_glossary.py      quy đổi từ đời thường → thuật ngữ luật ("đèn đỏ" → "đèn tín hiệu giao thông")
        ↓ LLM viết lại câu hỏi   thành truy vấn pháp lý, tự đủ nghĩa (chế độ QUERY_REWRITE)
        ↓ retrieval_hybrid.py
        ├─ BM25        → top 14 (từ khóa)
        ├─ FAISS       → top 14 (ngữ nghĩa)
        ├─ Chuẩn hóa min-max, cộng 0.5 × dense + 0.5 × BM25
        └─ Cross-encoder rerank → top 7 (+ tối đa 3 chunk có điểm sát chunk đầu)
        ↓ Prompt = nguyên tắc + lịch sử hội thoại + 7–10 tài liệu + câu hỏi
        ↓ LLM (Gemini trên cloud / Ollama local), temperature 0
Câu trả lời + nguồn (văn bản, Điều) + lưu ý về văn bản đã hết hiệu lực
```

---

## 2. Thành phần

### 2.1 Loader (`src/loader.py`)
`DirectoryLoader` đọc mọi file trong `data/`: `Docx2txtLoader` cho `.docx`, `UnstructuredWordDocumentLoader` cho `.doc` (bỏ qua nếu thiếu thư viện), `PyPDFLoader` cho `.pdf`.

### 2.2 Chunker (`src/chunker.py`)
Văn bản luật có cấu trúc **Điều → khoản (1., 2.) → điểm (a), b))**. Chunker bám theo cấu trúc đó:

1. **Tách theo Điều**: regex `Điều N.` **neo ở đầu dòng** (cho phép ngoặc kép mở trong luật sửa đổi). Không neo thì câu dẫn chiếu như "… khoản 3 Điều 71; t) Điều 72, Điều 73." bị tách nhầm thành một điều giả.
2. **Điều ≤ 1.000 ký tự** → giữ nguyên một chunk.
3. **Điều dài** → gom các khoản liền nhau thành chunk ≤ 1.000 ký tự, **không cắt ngang khoản**. Khoản nào vẫn dài quá thì tách tiếp bằng `RecursiveCharacterTextSplitter` (1.000 ký tự, overlap 300), mỗi mảnh sau mang theo dòng mở đầu của khoản (chủ thể + mức phạt).
4. **Đoạn ngắn < 300 ký tự** (thường chỉ là tiêu đề điều) luôn được gộp với phần kế tiếp, không đứng riêng.
5. **Gắn nhãn** đầu mỗi chunk: `[Nghị định 100/2019/NĐ-CP (…)] Điều 6. Xử phạt người điều khiển xe mô tô… (tiếp theo)`. Metadata có `law` và `article` để app hiển thị nguồn.

Tên văn bản lấy từ `DOC_LABELS` (theo số hiệu ghi ở đầu từng file); file mới chưa có trong bảng thì dùng "Văn bản số …" đọc từ dòng `Số:`. `DOC_NOTES` chứa lưu ý cho văn bản đã bị thay thế một phần (Nghị định 100/2019) hoặc không phải toàn văn (luật sửa đổi Bộ luật Hình sự).

Kết quả hiện tại: 3.394 chunk, không có chunk nào chỉ chứa tiêu đề hoặc mở đầu bằng một điểm a), b)… tách khỏi khoản.

### 2.3 Embedder (`src/embedder.py`)
Thứ tự chọn model: Custom Transformer (nếu có `models/custom-transformer/`) → fine-tuned E5 → base E5.
- Streamlit Cloud không có `models/` → tự tải fine-tuned E5 từ Hugging Face Hub (`Hoangy2005/vietnamese-legal-embedder`).
- Máy local có Custom Transformer 128 chiều → phải đặt `USE_FINETUNED=true` để dùng E5 khớp với index 768 chiều.

### 2.4 Vector store (`src/vector_store.py`)
FAISS (`IndexFlatL2`, brute force — đủ nhanh với vài nghìn chunk). Khi load, so số chiều index với embedding model và báo lỗi rõ ràng nếu lệch. `add_documents()` dùng cùng cách chunk để thêm văn bản mới mà không build lại.

### 2.5 Từ điển thuật ngữ (`src/legal_glossary.py`)
16 quy tắc regex đổi từ đời thường sang thuật ngữ có trong luật (mọi thuật ngữ đích đều đã kiểm tra là có trong `data/raw/`): "bằng lái" → "giấy phép lái xe", "rượu bia" → "nồng độ cồn", "đuổi việc" → "đơn phương chấm dứt hợp đồng lao động", "sổ đỏ" → "giấy chứng nhận quyền sử dụng đất"…

Ngoại lệ được giữ nguyên: câu có "đường sắt / đường ngang" (Điều 47 NĐ 100/2019 đúng là quy định về đèn đỏ ở đường ngang), "xe máy chuyên dùng", "nghỉ việc riêng".

### 2.6 Viết lại câu hỏi (`HybridQAChain.rewrite_query`)
LLM viết lại câu hỏi thành một truy vấn pháp lý, dùng lịch sử hội thoại để câu nối tiếp ("còn người đi bộ thì sao?") tự đủ nghĩa. Từ điển được áp dụng cả trước và sau bước này. Lỗi khi gọi LLM → dùng câu đã quy đổi.

`QUERY_REWRITE`: `auto` (mặc định) | `true` | `false`. Ở `auto`, Gemini viết lại mọi câu; Ollama chỉ viết lại câu nối tiếp, vì `qwen3:8b` viết lại câu độc lập kém hơn từ điển (từng bỏ mất chữ "đèn" làm rơi Điều 6).

### 2.7 Hybrid retriever (`src/retrieval_hybrid.py`)
- **BM25** (`rank-bm25`, tách từ theo khoảng trắng): mạnh với từ khóa chính xác — số điều, mức phạt.
- **Dense FAISS**: mạnh với cách diễn đạt khác từ ngữ.
- **Fusion**: chuẩn hóa min-max từng nguồn (khoảng cách FAISS được đảo chiều), `0.5 × dense + 0.5 × BM25`.
- **Cross-encoder** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` đọc đồng thời câu hỏi và đoạn văn để xếp hạng lại.
- **k thích ứng**: lấy top 7, cộng thêm tối đa 3 chunk có điểm cross-encoder cách chunk đứng đầu không quá 2.0. Câu hỏi nhiều đối tượng (vượt đèn đỏ: mỗi loại xe một Điều) cần nhiều hơn 7 chunk; câu hỏi thường gần như không bị ảnh hưởng.

### 2.8 LLM (`src/llm_chain.py`)
`LLM_PROVIDER` = `gemini` (mặc định khi có `GOOGLE_API_KEY`, model `gemini-3.5-flash-lite`) hoặc `ollama` (model local, mặc định `qwen3:8b`, `num_ctx=12288`, tắt thinking). Đổi model bằng `LLM_MODEL`. `temperature=0`, tối đa 2.048 token đầu ra. `is_transient_error()` nhận diện lỗi quá tải / giới hạn request (429, 503) cho app, API và evaluation.

### 2.9 Prompt
Nguyên tắc chính: trả lời thẳng vào câu hỏi; chỉ dùng thông tin trong tài liệu; trích dẫn gọn dạng "(khoản 2 Điều 8 Nghị định 100/2019/NĐ-CP)" và chỉ ghi khoản/điểm khi thấy rõ trong tài liệu; nếu tài liệu quy định khác nhau cho từng đối tượng thì liệt kê đủ mọi đối tượng; dùng lịch sử hội thoại cho câu hỏi tiếp theo.

### 2.10 Streamlit app (`src/app.py`)
Câu miễn trừ trách nhiệm song ngữ, 4 nút câu hỏi mẫu (rê chuột xem tiếng Anh), thanh bên liệt kê văn bản trong hệ thống, lưu ý khi nguồn là văn bản đã bị thay thế, mục "Nguồn tài liệu" kèm truy vấn đã viết lại, thông báo lỗi thân thiện (lỗi chi tiết ghi vào log).

### 2.11 REST API (`src/api.py`)
```
POST /chat
{ "message": "Còn người đi bộ thì sao?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}] }

→ 200 { "reply": "...", "search_query": "...",
        "sources": [{"law": "...", "article": "Điều 9", "excerpt": "..."}],
        "notes": ["Mức phạt giao thông đường bộ trong Nghị định 100/2019 ..."] }
→ 503 khi LLM provider quá tải, 500 khi lỗi khác, 422 khi request sai định dạng

GET /health → {"status": "ok"}
```
Handler là hàm đồng bộ để FastAPI chạy trong threadpool, không chặn event loop.

---

## 3. Mô hình embedding

### 3.1 Custom Transformer (`src/custom_transformer/`)
```
input_ids (≤128 token) → Token Embedding (vocab 2.672) + Positional Encoding (sin/cos)
→ TransformerEncoderLayer × 2 (4 heads, hidden 128, feed-forward 512, pre-LayerNorm, dropout 0.1)
→ Mean pooling (bỏ [PAD]) → L2 normalize → vector 128 chiều
```
- Tokenizer tự xây theo từ, lấy các từ phổ biến nhất trong văn bản luật (tối đa 8.000; thực tế 2.672 từ).
- 738.560 tham số (đếm từ `models/custom-transformer/model.pt`).
- Train: `TripletMarginLoss` (margin 0.3), AdamW (lr 3e-4, weight decay 1e-2), CosineAnnealingLR, 15 epoch, batch 32, CPU.

### 3.2 Fine-tuned multilingual-e5-base (`src/finetuned_e5/`)
- `intfloat/multilingual-e5-base`, 278.043.648 tham số, vector 768 chiều.
- sentence-transformers `TripletLoss` (margin 0.5), lr 2e-5, warmup 10%, 15 epoch, batch 32 → 1.710 bước (114 bước/epoch). Train trên Colab GPU (`train_colab.ipynb`), đẩy lên Hugging Face Hub.

---

## 4. Dữ liệu

| Tập | Số lượng | Nguồn |
|---|---|---|
| `all_questions.json` | 1.530 câu hỏi + đáp án trích nguyên văn | LLM sinh từ từng điều luật (`src/generate_test_cases.py`), kiểm tra ≥ 60% từ khóa của đáp án có trong điều luật |
| `train_set.json` / `test_set.json` | 1.224 / 306 | `split_dataset.py`, chia 80/20, seed 42 |
| `legal_triplets_dataset` | 3.642 triplet (1.214 câu hỏi × 3 hard negative) | `src/data_preparation.py`, hard negative lấy từ FAISS |
| `test_triplets_dataset` | 892 triplet | `generate_test_triplets.py`, negative ngẫu nhiên |

`src/augment_triplets.py` (paraphrase câu hỏi bằng LLM) có sẵn nhưng bộ 3.642 triplet hiện tại không chứa câu paraphrase.

**Vấn đề đã biết:** `train_custom.py` tạo triplet từ `test_set.json`, nên 232 câu hỏi của tập test nằm trong tập triplet huấn luyện (119/150 câu dùng để đánh giá). Cần tạo lại triplet chỉ từ `train_set.json` rồi đánh giá lại.

---

## 5. Đánh giá (`src/evaluation_enhanced.py`)

| Nhóm | Chỉ số | Cách tính |
|---|---|---|
| Retrieval | NDCG@7, MRR@10, Recall@7, Precision@7 | Một đoạn được coi là liên quan nếu trùng đủ từ với đáp án tham chiếu; chỉ chấm trên các đoạn đã lấy về |
| Sinh câu trả lời | RAGAS: faithfulness, answer relevancy, context precision, context recall | LLM chấm điểm là cùng provider với chatbot; gặp 429/503 thì chờ 30 giây, thử lại tối đa 3 lần |
| So với đáp án | Semantic similarity | Cosine giữa embedding (`paraphrase-multilingual-MiniLM-L12-v2`) của câu trả lời và đáp án |

`evaluation/evaluate_finetuned.py` chạy 150 câu đầu của tập test (RAGAS 50 câu); `evaluation/evaluate_custom.py` chạy 10 câu. Kết quả trong README được đo khi hệ thống còn dùng Llama trên Groq và trước các thay đổi chunking/retrieval ở trên — cần chạy lại.

---

## 6. Lỗi đã gặp và cách sửa

| Triệu chứng | Nguyên nhân gốc | Cách sửa |
|---|---|---|
| Hỏi "vượt đèn đỏ" nhưng trả lời theo Điều 47 (đường ngang đường sắt) | Cụm "đèn đỏ" chỉ xuất hiện nguyên văn ở Điều 47; các điều phạt vượt đèn đỏ viết "không chấp hành hiệu lệnh của đèn tín hiệu giao thông" | Từ điển thuật ngữ + LLM viết lại câu hỏi |
| Mức phạt xe đạp sai (200–300 nghìn thay vì 100–200 nghìn) | Overlap của splitter tách điểm đ) khỏi dòng "2. Phạt tiền…" và đặt cạnh "3. Phạt tiền từ 200.000…" | Chunk theo khoản, không cắt ngang khoản |
| Câu trả lời thiếu xe máy | Top-7 cố định; chunk Điều 6 đứng hạng 8, thấp hơn hạng 7 đúng 0,05 điểm | k thích ứng theo điểm cross-encoder |
| Trích dẫn "Điều 73" không tồn tại | Regex tách cả câu dẫn chiếu giữa dòng | Neo regex ở đầu dòng |
| 78 chunk chỉ có tiêu đề điều | Tiêu đề ngắn bị đẩy ra khi khoản kế tiếp dài | Gộp đoạn < 300 ký tự với phần kế tiếp |
| App lỗi khó hiểu khi chạy local | Index 768 chiều nhưng embedder chọn Custom Transformer 128 chiều | Kiểm tra số chiều khi load, gợi ý `USE_FINETUNED=true` |
| Build index bị crash trên macOS | Import `faiss` trước `torch` (xung đột OpenMP) | Import embedder trước FAISS |

---

## 7. Hạn chế hiện tại
- Mức phạt giao thông theo Nghị định 100/2019, đã được thay bằng Nghị định 168/2024 (sửa đổi bởi Nghị định 238/2026). Bản chính thức của hai nghị định mới trên vanban.chinhphu.vn chỉ có PDF scan, chưa có bản văn bản để đưa vào hệ thống; app hiện lưu ý này khi trả lời.
- `qwen3:8b` chạy local yếu với câu hỏi nhiều đối tượng; Gemini trả lời đầy đủ hơn.
- Đánh giá cần chạy lại sau khi sửa lỗi lẫn dữ liệu test vào train.
