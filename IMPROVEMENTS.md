# Cách khắc phục khi các chỉ số thấp — Chuẩn bị cho báo cáo

---

## 1. NDCG@7 thấp → Retrieval xếp hạng sai thứ tự

**Nghĩa là:** Chunk đúng tìm được nhưng nằm ở vị trí thấp (6, 7) thay vì vị trí 1, 2.

**Nguyên nhân:**
- Score fusion alpha=0.5 chưa cân bằng tốt giữa BM25 và Dense
- Cross-Encoder chưa đủ mạnh để rerank chính xác

**Cách khắc phục:**
```
1. Điều chỉnh alpha trong score fusion
   alpha=0.3 → nghiêng về BM25 (tốt với từ khóa chính xác như số điều)
   alpha=0.7 → nghiêng về Dense (tốt với câu hỏi ngữ nghĩa)
   → Thử nghiệm trên validation set để chọn alpha tốt nhất

2. Tăng số candidates trước khi rerank
   k_bm25=10, k_dense=10 → cross-encoder có nhiều lựa chọn hơn
   → Xác suất chunk đúng được rerank lên cao hơn

3. Dùng Cross-Encoder mạnh hơn
   Hiện tại: mmarco-mMiniLMv2-L12-H384-v1
   Nâng cấp: ms-marco-MiniLM-L-6-v2 hoặc cross-encoder/ms-marco-electra-base
```

---

## 2. MRR@10 thấp → Chunk đúng đầu tiên xuất hiện quá muộn

**Nghĩa là:** Câu hỏi nào đó phải xuống tận vị trí 3-4 mới thấy chunk đúng.

**Nguyên nhân:**
- Câu hỏi dùng từ đồng nghĩa → BM25 không bắt được
- Embedding chưa đủ tốt với domain pháp luật tiếng Việt

**Cách khắc phục:**
```
1. Query Expansion — sinh thêm biến thể câu hỏi trước khi retrieve
   "vi phạm tốc độ" → thêm "vượt tốc độ", "chạy quá tốc độ"
   → Retrieve nhiều góc độ hơn → chunk đúng xuất hiện sớm hơn

2. Fine-tune thêm embedding model với nhiều data hơn
   807 triplets → 2000+ triplets
   → Model hiểu đồng nghĩa pháp luật tốt hơn

3. Tăng k_dense để lưới tìm kiếm rộng hơn
   k_dense=7 → k_dense=14
```

---

## 3. Recall@7 thấp → Bỏ sót chunk đúng

**Nghĩa là:** Chunk chứa thông tin cần thiết không nằm trong top 7 trả về.

**Nguyên nhân:**
- k=7 quá nhỏ
- Chunk bị chia cắt làm thông tin nằm rải rác nhiều chunk

**Cách khắc phục:**
```
1. Tăng k lên 10-14
   retriever = hybrid_retriever.as_retriever(k=10)

2. Tăng chunk overlap để thông tin không bị cắt đứt
   chunk_size=1000, overlap=300 → overlap=400
   → Thông tin liên quan nằm trong cùng 1 chunk hơn

3. Tăng k_bm25 và k_dense trước khi fusion
   k_bm25=14, k_dense=14 → pool rộng hơn trước khi lọc
```

---

## 4. Precision@7 thấp → Trả về nhiều chunk không liên quan

**Nghĩa là:** Trong 7 chunk trả về có nhiều chunk "nhiễu" không liên quan câu hỏi.

**Nguyên nhân:**
- k=7 quá nhiều so với số chunk thực sự liên quan
- Cross-Encoder chưa lọc đủ mạnh

**Cách khắc phục:**
```
1. Giảm k xuống 5 (trade-off với Recall)
   k=7 → k=5 → ít chunk hơn → ít nhiễu hơn

2. Thêm score threshold cho Cross-Encoder
   Chỉ giữ chunk có cross-encoder score > 0.3
   → Tự động loại chunk kém liên quan

3. Metadata filtering — lọc theo nguồn văn bản
   Câu hỏi về giao thông → chỉ search trong Nghị định 100/2019
   → Loại trừ chunk từ Luật Lao động, Dân sự không liên quan
```

---

## 5. Faithfulness thấp → LLM bịa thêm thông tin (Hallucination)

**Nghĩa là:** Câu trả lời chứa thông tin không có trong tài liệu được cung cấp.

**Nguyên nhân:**
- LLM dùng kiến thức nền của mình thay vì chỉ dựa vào context
- Context không đủ thông tin → LLM tự điền vào chỗ trống

**Cách khắc phục:**
```
1. Siết chặt system prompt
   Thêm: "TUYỆT ĐỐI không thêm số liệu, mức phạt, điều khoản
          nếu không có trong TÀI LIỆU PHÁP LÝ bên dưới."
   Thêm: "Nếu không đủ thông tin, nói rõ phần nào không có."

2. Tăng k để LLM có đủ context
   k=7 → k=9 → nhiều thông tin hơn → ít phải "đoán" hơn

3. Giữ temperature=0 (đã làm)
   temperature=0 → LLM ít sáng tạo → ít hallucinate

4. Dùng mô hình LLM tốt hơn
   Llama-3.3-70b → GPT-4o hoặc Claude Sonnet
   → Tuân thủ instruction tốt hơn
```

---

## 6. Answer Relevancy thấp → Trả lời lạc đề

**Nghĩa là:** Câu trả lời không đúng chủ đề câu hỏi (hỏi giao thông trả lời lao động).

**Nguyên nhân:**
- Retrieval sai → LLM nhận context sai → trả lời sai chủ đề
- System prompt chưa đủ định hướng

**Cách khắc phục:**
```
1. Cải thiện Retrieval trước (Precision, NDCG)
   Answer Relevancy thấp thường do Retrieval sai → fix retrieval trước

2. Thêm instruction trong prompt
   "Chỉ trả lời đúng câu hỏi được hỏi, không mở rộng sang chủ đề khác."

3. Kiểm tra lại chunking
   Chunk quá lớn (1000 chars) → chứa nhiều chủ đề → embedding không đại diện tốt
   → Giảm chunk_size=600-800
```

---

## 7. Context Precision thấp → Chunk retrieve được chứa nhiều nội dung thừa

**Nghĩa là:** Trong 7 chunk trả về, nhiều chunk không cần thiết để trả lời câu hỏi.

**Nguyên nhân:**
- BM25 bắt được từ khóa nhưng chunk chứa nhiều điều khoản không liên quan
- Chunk size quá lớn → 1 chunk chứa nhiều chủ đề khác nhau

**Cách khắc phục:**
```
1. Giảm chunk_size để mỗi chunk tập trung 1 ý
   chunk_size=1000 → chunk_size=600
   → Mỗi chunk chứa 1-2 điều khoản thay vì 3-4

2. Tăng ngưỡng Cross-Encoder
   Chỉ giữ chunk có score cao → loại chunk liên quan xa

3. Giảm k từ 7 xuống 5
   Ít chunk hơn → ít chunk nhiễu hơn → Precision tăng

4. Cải thiện query với từ khóa pháp lý
   Thêm "Điều X", "Nghị định Y" vào query nếu có thể detect
```

---

## 8. Context Recall thấp → Thiếu thông tin để trả lời đầy đủ

**Nghĩa là:** Câu hỏi cần 3 ý nhưng chunk retrieve chỉ chứa 2 ý → trả lời thiếu.

**Nguyên nhân:**
- Thông tin bị chia rải rác nhiều chunk, k=7 không bắt được hết
- Overlap giữa các chunk quá nhỏ

**Cách khắc phục:**
```
1. Tăng chunk overlap
   overlap=300 → overlap=400
   → Thông tin liên tục không bị cắt đứt ở ranh giới chunk

2. Tăng k
   k=7 → k=10 → bắt được nhiều chunk hơn

3. Hierarchical chunking
   Chunk nhỏ (200 chars) để retrieve + chunk lớn (1000 chars) để đưa vào LLM
   → Tìm chính xác hơn, cung cấp context đầy đủ hơn

4. Tăng k_bm25 và k_dense
   k_bm25=14, k_dense=14 → pool lớn hơn trước khi rerank
```

---

## 9. BLEU thấp → Câu trả lời diễn đạt khác ground truth

**Nghĩa là:** Từ ngữ khác nhau dù nghĩa đúng.

**Nguyên nhân:**
- BLEU vốn không phù hợp với RAG chatbot (thiết kế cho dịch máy)
- LLM luôn paraphrase → khác từ ngữ với ground truth là bình thường

**Cách khắc phục:**
```
→ BLEU thấp (0.1-0.2) với RAG là BÌNH THƯỜNG, không cần lo

Nếu bắt buộc phải tăng:
1. Yêu cầu LLM trả lời ngắn gọn hơn
   Thêm vào prompt: "Trả lời súc tích, tối đa 2-3 câu."
   → Ít từ thừa → khớp ground truth nhiều hơn

2. Tăng số câu ground truth tham chiếu
   1 câu chuẩn → 3 cách diễn đạt khác nhau
   → BLEU tính max match → điểm cao hơn
```

---

## 10. ROUGE-L thấp → Câu trả lời thiếu cụm từ chung với ground truth

**Nghĩa là:** Chuỗi từ chung giữa câu trả lời và ground truth ngắn.

**Nguyên nhân:**
- Câu trả lời LLM dài hơn ground truth nhiều
- LLM dùng từ đồng nghĩa thay vì từ gốc trong tài liệu

**Cách khắc phục:**
```
1. Yêu cầu LLM trích dẫn nguyên văn từ tài liệu
   Thêm vào prompt: "Trích dẫn nguyên văn điều khoản khi có thể."
   → Dùng từ gốc → khớp ground truth hơn

2. Giảm max_tokens của LLM
   max_tokens=1024 → max_tokens=512
   → Câu ngắn hơn → ít "loãng" cụm từ chung

3. Tương tự BLEU — tăng ground truth đa dạng
```

---

## 11. Semantic Similarity thấp → Nghĩa câu trả lời khác ground truth

**Nghĩa là:** Embedding câu trả lời và ground truth nằm xa nhau trong không gian vector → nghĩa khác nhau.

**Nguyên nhân:**
- Câu trả lời đúng thông tin nhưng ngữ cảnh khác (quá dài, quá nhiều chi tiết phụ)
- Embedding model chưa đủ tốt với domain pháp luật

**Cách khắc phục:**
```
1. Fine-tune thêm với nhiều data hơn
   807 triplets → 2000+ triplets
   Thêm văn bản luật: Hình sự, Hôn nhân gia đình, Đất đai
   → Embedding hiểu domain pháp luật sâu hơn → similarity cao hơn

2. Dùng MultipleNegativesRankingLoss thay TripletLoss
   → Học hiệu quả hơn với dataset nhỏ
   → Embedding phân biệt nghĩa tốt hơn

3. Tăng epochs fine-tuning
   epochs=5 → epochs=8 (kèm early stopping)

4. Yêu cầu LLM trả lời đúng trọng tâm
   Thêm: "Trả lời thẳng vào câu hỏi, không thêm thông tin ngoài lề."
   → Câu trả lời gần ground truth hơn về nghĩa
```

---

## Tóm tắt nhanh để trả lời giảng viên

| Nếu hỏi về | Trả lời ngắn |
|---|---|
| NDCG/MRR thấp | Điều chỉnh alpha fusion, tăng candidates trước rerank |
| Recall thấp | Tăng k, tăng chunk overlap |
| Precision thấp | Giảm k, thêm score threshold cho Cross-Encoder |
| Faithfulness thấp | Siết prompt, tăng k để đủ context |
| Answer Relevancy thấp | Fix retrieval trước, thêm instruction vào prompt |
| Context Precision thấp | Giảm chunk_size, thêm score threshold |
| Context Recall thấp | Tăng overlap, tăng k |
| BLEU/ROUGE thấp | Metric không phù hợp RAG — bình thường, không cần lo |
| Semantic Similarity thấp | Thêm data training, dùng MultipleNegativesRankingLoss |
