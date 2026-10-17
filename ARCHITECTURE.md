# Kiến trúc hệ thống — Vietnamese Legal RAG Chatbot

Tài liệu này mô tả kiến trúc **hiện tại** của hệ thống (cập nhật 10/2026). Phần tổng quan bằng tiếng Anh nằm ở [README.md](README.md).

---

## 1. Tổng quan

Bố cục thư mục: `src/` chỉ chứa code mà app đang deploy chạy; `training/` tạo dữ liệu và train 2 model embedding; `evaluation/` đo chất lượng; `data/` gồm `raw/` (văn bản luật), `qa/` (bộ câu hỏi) và `triplets/` (dữ liệu train embedding). Mọi lệnh chạy từ thư mục gốc repo.

### Offline — chuẩn bị dữ liệu
```
data/raw/*.docx (10 văn bản luật)
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
        ↓ Cổng điểm: điểm rerank cao nhất < -4.0 → từ chối, không gọi LLM
        ↓ Prompt = nguyên tắc + lịch sử hội thoại + 7–10 tài liệu + câu hỏi
        ↓ LLM (Gemini trên cloud / Ollama local), temperature 0, stream từng token
Câu trả lời + nguồn (văn bản, Điều) + lưu ý về văn bản đã hết hiệu lực
```

`HybridQAChain` tách hai bước: `prepare()` (viết lại câu hỏi, tìm kiếm, cổng điểm; trả về câu trả lời có sẵn nếu là lời chào hoặc bị từ chối) và `stream_answer()` / `generate_answer()` (gọi LLM). App stream, còn API và evaluation dùng `invoke()` = `prepare()` + `generate_answer()`, nên kết quả hai bên giống nhau.

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

Văn bản sửa đổi kiểu "Điều 13. Sửa đổi, bổ sung Điều 42" + nội dung “Điều 42. …” trích nguyên văn được giữ chung một chunk (nhãn Điều 13 của văn bản sửa đổi); điều chỉ có tiêu đề như "Điều 10. (được bãi bỏ)" không bị gộp vào điều sau.

Kết quả hiện tại: 3.825 chunk, không có chunk nào chỉ chứa tiêu đề hoặc mở đầu bằng một điểm a), b)… tách khỏi khoản.

**Nghị định 168/2024 và 238/2026:** bản chính thức trên vanban.chinhphu.vn chỉ là PDF scan (không có lớp chữ). Bản Word lấy từ thuvienphapluat.vn được đối chiếu với bản scan bằng OCR (Apple Vision, `vi-VT`): khớp 473/473 con số ở Nghị định 168, 23/23 ở Nghị định 238; độ giống văn bản sau khi bỏ dấu 98,8% và 97%. Các chỗ khác nhau đều do OCR đọc sót (vệt đen trên bản scan, con dấu chữ ký số) và đã được kiểm tra trên ảnh trang gốc.

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
LLM viết lại câu hỏi thành một truy vấn pháp lý **bằng tiếng Việt có dấu**, dùng lịch sử hội thoại để câu nối tiếp ("còn người đi bộ thì sao?") tự đủ nghĩa. Nhờ vậy câu hỏi tiếng Anh ("What are the legal conditions for getting married?" → "Điều kiện kết hôn…") và tiếng Việt gõ không dấu cũng khớp được văn bản luật; câu trả lời được viết bằng ngôn ngữ của người hỏi. Từ điển được áp dụng cả trước và sau bước này. Lỗi khi gọi LLM → dùng câu đã quy đổi.

`QUERY_REWRITE`: `auto` (mặc định) | `true` | `false`. Ở `auto`, Gemini viết lại mọi câu; Ollama chỉ viết lại câu nối tiếp và câu không có dấu tiếng Việt (tiếng Anh, tiếng Việt không dấu), vì `qwen3:8b` viết lại câu tiếng Việt độc lập kém hơn từ điển (từng bỏ mất chữ "đèn" làm rơi Điều 6).

### 2.7 Hybrid retriever (`src/retrieval_hybrid.py`)
- **BM25** (`rank-bm25`, tách từ theo khoảng trắng): mạnh với từ khóa chính xác — số điều, mức phạt.
- **Dense FAISS**: mạnh với cách diễn đạt khác từ ngữ.
- **Fusion**: chuẩn hóa min-max từng nguồn (khoảng cách FAISS được đảo chiều), `0.5 × dense + 0.5 × BM25`.
- **Cross-encoder** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` đọc đồng thời câu hỏi và đoạn văn để xếp hạng lại.
- **k thích ứng**: lấy top 7, cộng thêm tối đa 3 chunk có điểm cross-encoder cách chunk đứng đầu không quá 2.0. Câu hỏi nhiều đối tượng (vượt đèn đỏ: mỗi loại xe một Điều) cần nhiều hơn 7 chunk; câu hỏi thường gần như không bị ảnh hưởng.
- **Lọc câu ngoài phạm vi** (`src/canned_replies.py`): lời chào / cảm ơn (cả câu) được trả lời sẵn, không chạy retrieval. Nếu điểm cross-encoder của đoạn đứng đầu < `MIN_RELEVANCE_SCORE` (mặc định −4.0), chain trả lời "chưa tìm thấy quy định liên quan" kèm danh sách văn bản, **không gọi LLM**.

  Ngưỡng được chọn từ dữ liệu: điểm đoạn đứng đầu của 306 câu test (p1 = −3.49, p5 = 0.32, trung vị 6.61) và 20 câu ngoài phạm vi (từ 0.23 với thuế TNCN đến −6.67 với nấu phở). Ở −4.0: từ chối 3/306 câu hợp lệ — cả 3 là ca retrieval vốn đã sai (vd. "tốc độ tối đa trong khu dân cư" không có đoạn đúng trong top 7) — và chặn 9/20 câu ngoài phạm vi rõ ràng. Câu ngoài phạm vi còn lọt qua (thường là luật khác: thuế, cư trú) vẫn bị prompt yêu cầu không trả lời ngoài tài liệu.

### 2.8 LLM (`src/llm_chain.py`)
`LLM_PROVIDER` = `gemini` (mặc định khi có `GOOGLE_API_KEY`, model `gemini-3.5-flash-lite`) hoặc `ollama` (model local, mặc định `qwen3:8b`, `num_ctx=12288`, tắt thinking). Đổi model bằng `LLM_MODEL`. `temperature=0`, tối đa 2.048 token đầu ra. `is_transient_error()` nhận diện lỗi quá tải / giới hạn request (429, 503) cho app, API và evaluation.

### 2.9 Prompt
Nguyên tắc chính: trả lời thẳng vào câu hỏi; chỉ dùng thông tin trong tài liệu; trích dẫn gọn dạng "(khoản 2 Điều 8 Nghị định 100/2019/NĐ-CP)" và chỉ ghi khoản/điểm khi thấy rõ trong tài liệu; nếu tài liệu quy định khác nhau cho từng đối tượng thì liệt kê đủ mọi đối tượng; dùng lịch sử hội thoại cho câu hỏi tiếp theo.

### 2.10 Streamlit app (`src/app.py`)
Câu miễn trừ trách nhiệm song ngữ, 4 nút câu hỏi mẫu tiếng Việt (rê chuột xem tiếng Anh) và 2 nút tiếng Anh, thanh bên liệt kê văn bản trong hệ thống, lưu ý khi nguồn là văn bản đã bị thay thế, mục "Nguồn tài liệu" kèm truy vấn đã viết lại, thông báo lỗi thân thiện (lỗi chi tiết ghi vào log).

Bảo vệ quota của demo công khai (mọi người dùng chung một API key):
- `DEMO_SESSION_LIMIT` (mặc định 20): số câu hỏi mỗi phiên trình duyệt; hết thì khóa ô nhập.
- `DEMO_DAILY_LIMIT` (mặc định 200): số câu hỏi *thực sự gọi LLM* mỗi ngày cho cả app (bộ đếm dùng chung qua `st.cache_resource`).
- Câu hỏi mở đầu phiên (không có lịch sử, vd. bấm nút mẫu) được cache 24 giờ, tối đa 500 câu: hỏi lại không gọi LLM và không tính vào giới hạn ngày. Cache tự quản lý (dict trong `st.cache_resource`) thay vì `st.cache_data`, vì `st.cache_data` cần trả về kết quả hoàn chỉnh nên không stream được. Chỉ cache chế độ văn bản luật, không bao giờ cache câu trả lời về tài liệu tải lên.

Câu trả lời hiện dần bằng `st.write_stream`. Lỗi giữa chừng (vd. 429 khi đang stream) được nối thành thông báo thân thiện vào cuối phần đã hiện, và câu đó không được cache. Mỗi tin nhắn lưu kèm nguồn, nên phần "Nguồn tài liệu" vẫn còn sau khi trang chạy lại.

### 2.10b Chế độ "Tài liệu của bạn" (`src/user_docs.py`, `src/document_upload.py`)
Nút chuyển ở đầu trang. Người dùng tải lên tối đa 3 tệp PDF / DOCX / TXT / MD (≤ 10 MB mỗi tệp, ≤ 400 chunk), hoặc bấm "Dùng tài liệu mẫu" (sổ tay nhân viên của công ty hư cấu trong `samples/`).
- Đọc chữ: `pypdf` (PDF), `docx2txt` (DOCX). Dưới 50 ký tự → báo lỗi "có thể là PDF scan" thay vì trả lời rỗng.
- Dùng lại chunker (nhãn chunk = tên tệp, chunk ≤ 400 ký tự vì các mục của tài liệu thường ngắn), embedder và cross-encoder đã tải sẵn; FAISS + BM25 dựng trong bộ nhớ, lưu ở `st.session_state` của phiên đó, khóa theo hash nội dung tệp. Không ghi xuống đĩa, không dùng chung giữa các phiên.
- Không viết lại câu hỏi (tài liệu thuộc mọi lĩnh vực). Prompt viết bằng tiếng Anh vì với prompt tiếng Việt, Gemini trả lời câu hỏi tiếng Anh bằng tiếng Việt.
- Cổng điểm `USER_DOCS_MIN_RELEVANCE_SCORE` = -6.0, lỏng hơn văn bản luật. Đo trên sổ tay mẫu: 10 câu có trong tài liệu (Anh + Việt) thấp nhất -4.39 (câu tiếng Việt hỏi tài liệu tiếng Anh); 6 câu ngoài tài liệu nằm từ -7.15 tới -0.77. Cross-encoder tách kém khi khác ngôn ngữ, nên cổng chỉ chặn câu rõ ràng không liên quan; các câu còn lại thì LLM tự nói "tài liệu không đề cập".
- Câu hỏi ở chế độ này tính chung vào giới hạn phiên và giới hạn ngày.

### 2.10c Giám sát (`src/observability.py`)
Tuỳ chọn, chỉ bật khi có `LANGFUSE_PUBLIC_KEY` và `LANGFUSE_SECRET_KEY` (Langfuse SDK v5, LangChain `CallbackHandler`). Mỗi câu hỏi là một trace gồm các bước `rewrite_query`, `retrieve` (kèm các đoạn tìm được) và `answer` (token, độ trễ), gắn session của trình duyệt, tag (`law` / `uploaded-docs` / `api`, provider) và metadata: truy vấn tìm kiếm, điểm rerank cao nhất, số nguồn, có phải câu trả lời có sẵn không, có lấy từ cache không. Câu hỏi về tài liệu tải lên chỉ được trace khi đặt `LANGFUSE_TRACE_UPLOADS=true`. Lỗi của Langfuse không làm hỏng câu trả lời: việc gửi trace chạy ở luồng nền.

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
Handler là hàm đồng bộ để FastAPI chạy trong threadpool, không chặn event loop. Khi bật Langfuse, mỗi request cũng là một trace (tag `api`).

---

## 3. Mô hình embedding

### 3.1 Custom Transformer (`src/custom_transformer/`, train: `training/train_custom.py`)
```
input_ids (≤128 token) → Token Embedding (vocab 2.672) + Positional Encoding (sin/cos)
→ TransformerEncoderLayer × 2 (4 heads, hidden 128, feed-forward 512, pre-LayerNorm, dropout 0.1)
→ Mean pooling (bỏ [PAD]) → L2 normalize → vector 128 chiều
```
- Tokenizer tự xây theo từ, lấy các từ phổ biến nhất trong văn bản luật (tối đa 8.000; thực tế 2.672 từ).
- 738.560 tham số (đếm từ `models/custom-transformer/model.pt`).
- Train: `TripletMarginLoss` (margin 0.3), AdamW (lr 3e-4, weight decay 1e-2), CosineAnnealingLR, 15 epoch, batch 32, CPU.

### 3.2 Fine-tuned multilingual-e5-base (`training/train_finetuned.py`, `training/finetuned_e5_trainer.py`)
- `intfloat/multilingual-e5-base`, 278.043.648 tham số, vector 768 chiều.
- sentence-transformers `TripletLoss` (margin 0.5), lr 2e-5, warmup 10%, 15 epoch, batch 32 → 1.710 bước (114 bước/epoch). Train trên Colab GPU (`training/train_colab.ipynb`), đẩy lên Hugging Face Hub.

---

## 4. Dữ liệu

| Tập | Số lượng | Nguồn |
|---|---|---|
| `data/qa/all_questions.json` | 1.530 câu hỏi + đáp án trích nguyên văn | LLM sinh từ từng điều luật (`training/generate_test_cases.py`), kiểm tra ≥ 60% từ khóa của đáp án có trong điều luật |
| `data/qa/train_set.json` / `test_set.json` | 1.224 / 306 | `training/split_dataset.py`, chia 80/20, seed 42 |
| `data/triplets/legal_triplets_dataset` | 3.642 triplet (1.214 câu hỏi × 3 hard negative) | `training/data_preparation.py`, hard negative lấy từ FAISS |
| `data/triplets/test_triplets_dataset` | 892 triplet | `training/generate_test_triplets.py`, negative ngẫu nhiên |

`training/augment_triplets.py` (paraphrase câu hỏi bằng LLM) có sẵn nhưng bộ 3.642 triplet hiện tại không chứa câu paraphrase.

**Vấn đề đã biết:** `training/train_custom.py` tạo triplet từ `test_set.json`, nên 232/306 câu hỏi của tập test nằm trong tập triplet huấn luyện. Cách xử lý hiện tại (không cần train lại E5): `evaluation/make_clean_test_set.py` lọc ra **74 câu chưa từng xuất hiện khi train** → `test_set_clean.json`, dùng làm tập đánh giá mặc định. 8/74 câu có đoạn luật đáp án trùng với đoạn luật của một câu train khác (đoạn luật là nội dung kho văn bản nên chấp nhận được; câu hỏi thì không trùng).

---

## 5. Đánh giá (`evaluation/rag_evaluator.py`)

| Nhóm | Chỉ số | Cách tính |
|---|---|---|
| Retrieval | NDCG@7, MRR@10, Recall@7, Precision@7 | Một đoạn được coi là liên quan nếu trùng đủ từ với đáp án tham chiếu; chỉ chấm trên các đoạn đã lấy về |
| Sinh câu trả lời | RAGAS: faithfulness, answer relevancy, context precision, context recall | Giám khảo mặc định cùng provider với chatbot; `EVAL_JUDGE_PROVIDER=ollama` để chấm bằng model local trong khi Gemini vẫn sinh câu trả lời. Ghi lại số mẫu chấm được (giám khảo trả JSON lỗi → bỏ qua). Gặp 429/503 thì chờ 30 giây, thử lại tối đa 3 lần |
| So với đáp án | Semantic similarity | Cosine giữa embedding (`paraphrase-multilingual-MiniLM-L12-v2`) của câu trả lời và đáp án |

`evaluation/evaluate_finetuned.py` mặc định chạy toàn bộ `test_set_clean.json` (tùy chọn `--limit`, `--ragas-limit`, `--test-set`) và lưu điểm kèm cấu hình chạy (LLM trả lời, giám khảo, số câu, ngày) vào `evaluation/results_finetuned.json`. `evaluation/evaluate_custom.py` chạy 10 câu.

Kết quả hiện tại (57 câu, 17/10/2026, Gemini trả lời, `qwen3:8b` chấm): NDCG@7 0,9905 · MRR@10 1,0 · Recall@7 0,9908 · Precision@7 0,9198; faithfulness 0,8377 (55/57 câu, 2 câu giám khảo trả JSON lỗi) · answer relevancy 0,7891 · context precision 0,8756 · context recall 0,9123; semantic similarity 0,7278. RAGAS gọi giám khảo 635 lần (≈ 11 lần/câu, không lỗi nào), mất 117 phút trên M1 Pro (≈ 2 phút/câu). Kết quả cũ thời Groq (tập test lẫn dữ liệu train, trước các thay đổi chunking/retrieval) chỉ còn trong phần thu gọn của README, không tái lập được.

---

### Bộ câu hỏi "đáp án vàng" (`evaluation/golden_set.json`)
25 câu phổ biến, đáp án trích từ văn bản luật (kèm điều/khoản/điểm): giao thông 10 (Nghị định 168/2024 và 238/2026), lao động 5, doanh nghiệp 3, dân sự 3, hôn nhân và gia đình 2, và 2 câu hỏi bằng tiếng Anh. `evaluation/golden_check.py` chấm tự động: phải có đủ các ý chính (khoảng tiền phạt, chấp nhận "18.000.000" hoặc "18 triệu"; số ngày, độ tuổi…), trích đúng Điều và văn bản; câu giao thông không được chứa mức phạt cũ của Nghị định 100/2019. Chạy: `python evaluation/run_golden_set.py` → `evaluation/results_golden.json` (kèm kết quả theo lĩnh vực). Kết quả hiện tại với Gemini 3.5 Flash-Lite: **25/25**.

Hai lần chạy đầu của 10 câu giao thông (8/10, 9/10) đã phát hiện 3 lỗi thật, sửa xong mới đạt 10/10 — xem 3 dòng cuối bảng dưới.

Tập đánh giá tự động `test_set_clean.json` (57 câu) bỏ thêm 17 câu có đáp án theo Nghị định 100/2019, vì câu trả lời đúng theo Nghị định 168 sẽ bị chấm lệch đáp án cũ. Khi đánh giá bằng Gemini, mỗi câu cách nhau ~8 giây (`EVAL_REQUEST_INTERVAL`) để không vượt 15 request/phút của free tier.

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
| Trả lời mức phạt giao thông đã hết hiệu lực | Nghị định 168/2024 chỉ có bản scan chính thức | Bản Word + đối chiếu từng con số với bản scan bằng OCR; prompt ưu tiên Nghị định 168 theo Điều 53, 54 của chính nghị định này |
| Đáp án vàng: "nồng độ cồn xe máy" đúng số tiền nhưng trích Nghị định 100 | Hai nghị định có điều gần giống nhau; chunk Nghị định 100 chiếm hết 14+14 chỗ ứng viên của BM25/FAISS, chunk đúng của Nghị định 168 không vào được danh sách ứng viên (lọc sau rerank không cứu được) | Gắn cờ `superseded` cho Chương II (Điều 5–38, đường bộ) Nghị định 100 và loại ngay từ bước lấy ứng viên, trừ khi truy vấn nhắc "100/2019"; Chương III (đường sắt) vẫn tìm bình thường |
| Đáp án vàng: "vượt đèn đỏ bị trừ mấy điểm" → "tài liệu không có" | Mức trừ điểm nằm ở khoản riêng ("điểm b khoản 9 … trừ 04 điểm"), khác chunk với hành vi | Khi chunk, gắn "[Bị trừ 04 điểm … theo điểm b khoản 16]" vào ngay dòng của từng hành vi; đọc tham chiếu lần lượt để xử lý cả "điểm b khoản 7, điểm c khoản 9" và "khoản 10" |
| Đánh giá bằng Gemini bị lỗi 429 | Free tier 15 request/phút, mỗi câu 2 request | Giãn cách ~8 giây/câu khi đánh giá |
| Tài liệu tải lên: hỏi tiếng Việt về tệp tiếng Anh → "tài liệu không có thông tin" | Chunk 1.000 ký tự gom 4–5 mục, cross-encoder chấm -5.49 khi khác ngôn ngữ, dưới ngưỡng -4.0 | Chunk 400 ký tự cho tài liệu tải lên, ngưỡng riêng -6.0 (đo trên 16 câu) |
| Tài liệu tải lên: hỏi tiếng Anh nhưng trả lời tiếng Việt, chép cả dòng "[tên tệp]" | Prompt viết bằng tiếng Việt | Prompt tiếng Anh, quy tắc ngôn ngữ ghi rõ, cấm chép dòng tiêu đề |
| Chuyển chế độ qua lại làm mất tài liệu đã tải | Streamlit bỏ trạng thái của widget không hiển thị nên uploader trở về rỗng; code coi "uploader rỗng" là người dùng đã xoá tệp | Chỉ xoá index trong `on_change` của uploader, tức là khi người dùng thật sự bỏ tệp |
| `embedder` chọn sai model khi import theo thứ tự khác | `USE_FINETUNED` nằm trong `.env`, nhưng `.env` chỉ được nạp khi `llm_chain` được import trước | `embedder.py` tự gọi `load_dotenv()` |

---

## 7. Hạn chế hiện tại
- Nghị định 168/2024 và 238/2026 là hai văn bản riêng (chưa có văn bản hợp nhất): câu trả lời đúng khi tìm được cả điều gốc lẫn điều sửa đổi.
- Câu hỏi về vi phạm *trước* 01/01/2025 chỉ dùng Nghị định 100/2019 khi người hỏi nhắc tới chính nghị định này.
- Mảng giao thông chỉ được kiểm bằng 10 câu đáp án vàng; tập đánh giá tự động (57 câu) không còn câu giao thông. Bộ đáp án vàng 25 câu chỉ kiểm các câu đã biết đáp án, không thay cho đánh giá trên câu hỏi mới.
- Chế độ "Tài liệu của bạn" chưa hỗ trợ OCR cho PDF scan. Cổng điểm của chế độ này mới chỉ được đo trên một tài liệu mẫu. Tài liệu dài hơn 400 chunk (mỗi chunk ≤ 400 ký tự) bị từ chối để việc embed trên CPU của bản demo không quá lâu.
- `qwen3:8b` chạy local yếu với câu hỏi nhiều đối tượng; Gemini trả lời đầy đủ hơn.
- E5 vẫn được fine-tune trên tập triplet có lẫn câu test; đánh giá tránh vấn đề này bằng `test_set_clean.json`, nhưng muốn dùng lại cả 306 câu thì cần tạo lại triplet chỉ từ `train_set.json` và train lại.
