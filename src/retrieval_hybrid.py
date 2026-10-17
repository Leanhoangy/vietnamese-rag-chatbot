"""
Enhanced retrieval with hybrid search: BM25 + Dense + Cross-Encoder Re-ranking
Combines sparse (BM25) and dense (FAISS) retrieval for better results
"""

import os
from typing import Any, Dict, Iterator, List, Protocol, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi


class DocumentLike(Protocol):
    """Minimal document contract used by the retriever."""

    page_content: str
    metadata: Dict[str, Any]

try:
    from .canned_replies import is_small_talk, out_of_scope_reply, small_talk_reply
    from .legal_glossary import apply_glossary, has_vietnamese_diacritics
    from .llm_chain import LLM_PROVIDER, model
    from .vector_store import vectorstore
    from .chunker import splits
except ImportError:
    from canned_replies import is_small_talk, out_of_scope_reply, small_talk_reply
    from legal_glossary import apply_glossary, has_vietnamese_diacritics
    from llm_chain import LLM_PROVIDER, model
    from vector_store import vectorstore
    from chunker import splits

# Import cross-encoder for re-ranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("Warning: CrossEncoder not available. Install with: pip install sentence-transformers")


class HybridRetriever:
    """Hybrid retriever combining BM25, dense vectors, and cross-encoder re-ranking"""
    
    def __init__(
        self,
        documents: List[DocumentLike],
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        store=None,
        cross_encoder=None,
    ):
        """
        Initialize hybrid retriever

        Args:
            documents: List of LangChain Document objects
            use_cross_encoder: Whether to use cross-encoder re-ranking
            cross_encoder_model: Model for cross-encoder
            store: FAISS store for dense search (default: the legal corpus index)
            cross_encoder: already-loaded CrossEncoder to reuse (vd. index tài liệu người dùng tải lên)
        """
        self.documents = documents
        self.doc_texts = [doc.page_content for doc in documents]
        self.store = store if store is not None else vectorstore
        
        # Initialize BM25 (sparse retriever)
        print("Initializing BM25 retriever...")
        tokenized_docs = [doc.split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        # Chunk của văn bản đã bị thay thế (xem chunker.SUPERSEDED_ARTICLES) — loại khỏi ứng viên mặc định
        self.superseded = np.array([bool(doc.metadata.get("superseded")) for doc in documents])
        
        # Initialize cross-encoder if available
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE
        if cross_encoder is not None:
            self.cross_encoder = cross_encoder
        elif self.use_cross_encoder:
            print(f"Loading cross-encoder: {cross_encoder_model}")
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model, device="cpu")
            except Exception as exc:
                print(f"Warning: could not load cross-encoder, falling back to BM25 + dense only: {exc}")
                self.cross_encoder = None
                self.use_cross_encoder = False
        else:
            self.cross_encoder = None

    @staticmethod
    def _document_id(doc: DocumentLike) -> str:
        """Build a stable ID so duplicate prefixes don't collapse distinct chunks."""
        source = doc.metadata.get("source", "")
        start_index = doc.metadata.get("start_index", -1)
        return f"{source}:{start_index}:{hash(doc.page_content)}"

    @staticmethod
    def _normalize_scores(
        results: List[Tuple[DocumentLike, float]],
        higher_is_better: bool,
    ) -> List[Tuple[DocumentLike, float]]:
        if not results:
            return []

        raw_scores = [score for _, score in results]
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if max_score == min_score:
            return [(doc, 1.0) for doc, _ in results]

        normalized = []
        for doc, score in results:
            value = (score - min_score) / (max_score - min_score)
            if not higher_is_better:
                value = 1.0 - value
            normalized.append((doc, float(value)))
        return normalized
    
    def retrieve_bm25(self, query: str, k: int = 5, include_superseded: bool = False) -> List[Tuple[DocumentLike, float]]:
        """Retrieve using BM25 (sparse lexical search)"""
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        if not include_superseded:
            scores = np.where(self.superseded, 0.0, scores)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def retrieve_dense(self, query: str, k: int = 5, include_superseded: bool = False) -> List[Tuple[DocumentLike, float]]:
        """Retrieve using dense vectors (semantic search)"""
        if include_superseded:
            return self.store.similarity_search_with_score(query, k=k)
        return self.store.similarity_search_with_score(
            query, k=k, filter=lambda metadata: not metadata.get("superseded"), fetch_k=k * 4,
        )
    
    def rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Tuple[DocumentLike, float]],
        k: int = 3,
    ) -> List[Tuple[DocumentLike, float]]:
        """Re-rank candidates using cross-encoder"""
        if not self.cross_encoder or not candidates:
            return candidates[:k]
        
        # Prepare pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Combine documents with cross-encoder scores
        reranked = list(zip(candidates, scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return [(doc, float(score)) for (doc, _), score in reranked[:k]]
    
    def hybrid_search(
        self,
        query: str,
        k_bm25: int = 5,
        k_dense: int = 5,
        k_rerank: int = 3,
        rerank: bool = True,
        alpha: float = 0.5,  # Weight for combining BM25 and dense scores
        include_superseded: bool = False,
    ) -> List[Tuple[DocumentLike, float]]:
        """
        Hybrid search combining BM25, dense, and optionally cross-encoder
        
        Args:
            query: User query
            k_bm25: Number of results from BM25
            k_dense: Number of results from dense retrieval
            k_rerank: Final number of results after re-ranking
            rerank: Whether to use cross-encoder re-ranking
            alpha: Weight for combining scores (0 = pure BM25, 1 = pure dense)
        
        Returns:
            List of (Document, score) sorted by final score
        """
        # Retrieve from both retrievers
        bm25_results = self.retrieve_bm25(query, k=k_bm25, include_superseded=include_superseded)
        dense_results = self.retrieve_dense(query, k=k_dense, include_superseded=include_superseded)
        
        # BM25 scores: higher is better. FAISS scores are distances: lower is better.
        bm25_results_norm = self._normalize_scores(bm25_results, higher_is_better=True)
        dense_results_norm = self._normalize_scores(dense_results, higher_is_better=False)
        
        # Combine results
        combined = {}
        for doc, score in bm25_results_norm:
            doc_id = self._document_id(doc)
            combined[doc_id] = {"doc": doc, "bm25_score": score, "dense_score": 0}
        
        for doc, score in dense_results_norm:
            doc_id = self._document_id(doc)
            if doc_id in combined:
                combined[doc_id]["dense_score"] = score
            else:
                combined[doc_id] = {"doc": doc, "bm25_score": 0, "dense_score": score}
        
        # Fuse scores
        fused_results = []
        for doc_info in combined.values():
            bm25_score = doc_info["bm25_score"]
            dense_score = doc_info["dense_score"]
            fused_score = alpha * dense_score + (1 - alpha) * bm25_score
            fused_results.append((doc_info["doc"], fused_score))
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        # Re-rank if enabled
        if rerank and self.use_cross_encoder and len(fused_results) > 0:
            # Take more candidates before re-ranking for better selection
            candidates = fused_results[:max(k_rerank * 2, len(fused_results))]
            fused_results = self.rerank_with_cross_encoder(query, candidates, k=k_rerank)
        else:
            fused_results = fused_results[:k_rerank]
        
        return fused_results
    
    def as_retriever(self, k: int = 3, **kwargs):
        """Return a LangChain-compatible BaseRetriever wrapper."""
        return LangChainHybridRetriever(
            hybrid_retriever=self,
            k=k,
            alpha=kwargs.get("alpha", 0.5),
        )


class LangChainHybridRetriever(BaseRetriever):
    """Adapter so the custom hybrid retriever works with current LangChain."""

    hybrid_retriever: HybridRetriever
    k: int = 3
    alpha: float = 0.5
    # k thích ứng: lấy thêm tối đa max_extra chunk nếu điểm cross-encoder sát chunk đứng đầu.
    # Câu hỏi nhiều đối tượng (vd. vượt đèn đỏ: ô tô, xe máy, xe đạp... mỗi đối tượng một Điều)
    # có thể cần hơn k chunk; câu hỏi thường ít khi có chunk ngoài top-k sát điểm như vậy.
    max_extra: int = 3
    score_margin: float = 2.0

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        results = self.hybrid_retriever.hybrid_search(
            query,
            k_bm25=self.k * 2,
            k_dense=self.k * 2,
            k_rerank=self.k + self.max_extra,
            rerank=True,
            alpha=self.alpha,
            # Phần đường bộ NĐ 100/2019 đã bị NĐ 168/2024 thay thế: chỉ tìm khi người hỏi nhắc tới chính NĐ 100
            include_superseded="100/2019" in query,
        )
        if self.hybrid_retriever.use_cross_encoder and results:
            top_score = results[0][1]
            extra = [r for r in results[self.k:] if r[1] >= top_score - self.score_margin]
            results = results[:self.k] + extra
            # Bản sao kèm điểm cross-encoder (không sửa Document dùng chung) để chain lọc câu ngoài phạm vi
            return [Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": score})
                    for doc, score in results]
        return [doc for doc, _ in results[:self.k]]


class HybridQAChain:
    """QA chain using hybrid retrieval"""
    
    def __init__(self, hybrid_alpha: float = 0.5):
        """hybrid_alpha: trọng số khi cộng điểm (0 = chỉ BM25, 1 = chỉ dense)."""
        self.hybrid_alpha = hybrid_alpha
        self.hybrid_retriever = HybridRetriever(splits, use_cross_encoder=CROSS_ENCODER_AVAILABLE)
        retriever = self.hybrid_retriever.as_retriever(k=7, alpha=hybrid_alpha)

        prompt_template = """Bạn là trợ lý tư vấn pháp luật Việt Nam.

NGUYÊN TẮC:
- Nếu câu hỏi là lời chào (ví dụ: "hello", "xin chào", "hi"...), chào lại ngắn gọn và hỏi người dùng cần tư vấn gì.
- Nếu câu hỏi liên quan pháp luật: trả lời THẲNG VÀO NỘI DUNG, không chào hỏi thêm. Trích dẫn chính xác điều khoản, mức phạt từ tài liệu.
- Mỗi tài liệu mở đầu bằng [Tên văn bản] và số Điều. Trích dẫn ngắn gọn ngay sau ý liên quan, dạng "(Điều 6 Nghị định 100/2019/NĐ-CP)" hoặc "(khoản 2 Điều 8 Bộ luật Lao động 2019)": dùng tên văn bản bỏ phần trong ngoặc, không chép nguyên dòng [...] ở đầu tài liệu, không tự suy ra số Điều.
- Chỉ ghi khoản, điểm khi thấy rõ số khoản, điểm đó trong nội dung tài liệu; không chắc thì chỉ ghi số Điều.
- Nếu tài liệu có thông tin liên quan: trả lời những gì có, không nói "không đủ thông tin".
- Chỉ nói "Tài liệu không cung cấp thông tin" khi hoàn toàn không tìm thấy gì liên quan.
- Không bịa thông tin ngoài tài liệu.
- Nếu tài liệu quy định khác nhau cho từng đối tượng (vd. ô tô, xe máy, xe đạp, người đi bộ; người lao động, người sử dụng lao động), liệt kê ĐẦY ĐỦ mọi đối tượng có trong tài liệu, mỗi đối tượng một dòng.
- Xử phạt vi phạm giao thông đường bộ: dùng Nghị định 168/2024/NĐ-CP (và phần sửa đổi tại Nghị định 238/2026/NĐ-CP nếu tài liệu có). KHÔNG dùng mức phạt đường bộ của Nghị định 100/2019/NĐ-CP, trừ khi người hỏi nói rõ vi phạm xảy ra trước 01/01/2025 (Điều 53, 54 Nghị định 168/2024).
- Trả lời bằng đúng ngôn ngữ của CÂU HỎI. Hỏi tiếng Anh thì trả lời tiếng Anh, dịch tên văn bản sang tiếng Anh và giữ số hiệu, vd. "(Article 6, Decree 168/2024/NĐ-CP)", "(Article 113, Labor Code 2019)".
- Dựa vào LỊCH SỬ HỘI THOẠI để hiểu ngữ cảnh các câu hỏi tiếp theo.

LỊCH SỬ HỘI THOẠI:
{chat_history}

TÀI LIỆU PHÁP LÝ:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["chat_history", "context", "question"]
        )

        # Câu hỏi đời thường ("bằng lái", "đuổi việc"...) thường không khớp thuật ngữ trong luật
        # → LLM viết lại thành truy vấn pháp lý trước khi retrieve
        rewrite_template = """Viết lại câu hỏi của người dùng thành MỘT câu truy vấn để tìm kiếm trong văn bản pháp luật Việt Nam.

YÊU CẦU:
- THAY từ ngữ đời thường bằng thuật ngữ pháp lý tương ứng như trong luật, nghị định; không giữ lại từ đời thường đã được thay. Ví dụ: "bằng lái" → "giấy phép lái xe"; "xe máy" → "xe mô tô, xe gắn máy"; "bị đuổi việc" → "người sử dụng lao động đơn phương chấm dứt hợp đồng lao động".
- Nếu câu hỏi phụ thuộc LỊCH SỬ HỘI THOẠI (ví dụ "còn ô tô thì sao?"), bổ sung chủ đề đang nói tới để truy vấn tự đủ nghĩa.
- Giữ nguyên ý hỏi. Không trả lời câu hỏi, không thêm thông tin mới.
- Câu hỏi bằng tiếng Anh (hoặc ngôn ngữ khác) hay tiếng Việt không dấu: truy vấn LUÔN viết bằng tiếng Việt có dấu, dùng thuật ngữ pháp lý Việt Nam (văn bản luật chỉ có tiếng Việt).
- Nếu là lời chào hoặc không liên quan pháp luật, trả về nguyên văn câu hỏi.
- Chỉ trả về một dòng truy vấn, không giải thích.

LỊCH SỬ HỘI THOẠI:
{chat_history}

CÂU HỎI: {question}

TRUY VẤN:"""

        def format_docs(docs):
            return "\n\n".join([
                f"[Tài liệu {i+1}]:\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])

        self.format_docs = format_docs
        self.prompt = prompt
        self.rewrite_prompt = PromptTemplate(
            template=rewrite_template,
            input_variables=["chat_history", "question"],
        )
        # QUERY_REWRITE: "auto" (mặc định) | "true" | "false".
        # auto: Gemini viết lại mọi câu. Model local nhỏ (Ollama) viết lại câu hỏi độc lập còn kém hơn
        # từ điển (vd. qwen3:8b bỏ mất chữ "đèn"), nên chỉ dùng LLM cho câu nối tiếp cần lịch sử hội thoại.
        self.query_rewrite_mode = os.getenv("QUERY_REWRITE", "auto").strip().lower()
        # Điểm cross-encoder của đoạn đứng đầu thấp hơn ngưỡng → coi là ngoài phạm vi.
        # -4.0 đo trên 306 câu test + 20 câu ngoài phạm vi: từ chối 1% câu hợp lệ (đều là ca retrieval sai)
        # và 9/20 câu ngoài phạm vi rõ ràng (nấu ăn, code, giá vàng...).
        self.min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", "-4.0"))
        self.model = model
        self.retriever = retriever

    def _should_rewrite(self, query: str, chat_history: str) -> bool:
        if self.query_rewrite_mode in ("true", "false"):
            return self.query_rewrite_mode == "true"
        # Câu tiếng Anh / tiếng Việt không dấu phải được viết lại sang tiếng Việt có dấu thì BM25 mới khớp
        return LLM_PROVIDER != "ollama" or bool(chat_history.strip()) or not has_vietnamese_diacritics(query)

    def rewrite_query(self, query: str, chat_history: str = "", config: Dict | None = None) -> str:
        """Quy đổi từ đời thường theo từ điển, rồi LLM viết lại thành truy vấn pháp lý; lỗi thì dùng câu đã quy đổi."""
        query = apply_glossary(query)
        if not self._should_rewrite(query, chat_history):
            return query
        try:
            raw = (self.rewrite_prompt | self.model | StrOutputParser()).invoke({
                "chat_history": chat_history or "(không có)",
                "question": query,
            }, config={**(config or {}), "run_name": "rewrite_query"})
        except Exception as e:
            print(f"Query rewrite failed, using original query: {e}")
            return query

        lines = [line.strip().strip('"“”') for line in raw.splitlines() if line.strip()]
        rewritten = lines[0] if lines else ""
        # Áp dụng lại từ điển: LLM đôi khi thêm ngược từ đời thường vào truy vấn
        return apply_glossary(rewritten) if 0 < len(rewritten) <= 300 else query

    def prepare(self, query: str, chat_history: str = "", config: Dict | None = None) -> Dict:
        """Mọi bước trước khi sinh câu trả lời: viết lại câu hỏi, tìm kiếm, cổng điểm liên quan.

        "result" khác None → câu trả lời có sẵn (lời chào, từ chối), không cần gọi LLM sinh câu trả lời.
        Ngược lại "prompt_inputs" dùng cho stream_answer / generate_answer.
        """
        if is_small_talk(query):
            return {"result": small_talk_reply(query), "source_documents": [], "search_query": query}

        search_query = self.rewrite_query(query, chat_history, config)
        docs = self.retriever.invoke(search_query, config={**(config or {}), "run_name": "retrieve"})

        # Không đoạn luật nào đủ liên quan → từ chối thay vì để LLM tự trả lời (dễ bịa), không tốn lượt gọi LLM
        top_score = docs[0].metadata.get("rerank_score") if docs else None
        if not docs or (top_score is not None and top_score < self.min_relevance_score):
            return {"result": out_of_scope_reply(query), "source_documents": [], "search_query": search_query}

        return {
            "result": None,
            "source_documents": docs,
            "search_query": search_query,
            "prompt_inputs": {
                "chat_history": chat_history,
                "context": self.format_docs(docs),
                "question": query,
            },
        }

    def _answer_chain(self):
        return self.prompt | self.model | StrOutputParser()

    def stream_answer(self, prepared: Dict, config: Dict | None = None) -> Iterator[str]:
        return self._answer_chain().stream(prepared["prompt_inputs"], config={**(config or {}), "run_name": "answer"})

    def generate_answer(self, prepared: Dict, config: Dict | None = None) -> str:
        return self._answer_chain().invoke(prepared["prompt_inputs"], config={**(config or {}), "run_name": "answer"})

    def invoke(self, query_dict: Dict) -> Dict:
        """Invoke the QA chain (compatible with LangChain interface)"""
        prepared = self.prepare(query_dict.get("query", ""), query_dict.get("chat_history", ""))
        if prepared["result"] is None:
            prepared["result"] = self.generate_answer(prepared)
        prepared.pop("prompt_inputs", None)
        return prepared


print("Initializing hybrid retrieval (BM25 + dense + cross-encoder)...")
qa_chain = HybridQAChain(hybrid_alpha=0.5)
