"""
Enhanced retrieval with hybrid search: BM25 + Dense + Cross-Encoder Re-ranking
Combines sparse (BM25) and dense (FAISS) retrieval for better results
"""

import os
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi


class DocumentLike(Protocol):
    """Minimal document contract used by the retriever."""

    page_content: str
    metadata: Dict[str, Any]

try:
    from .llm_chain import model
    from .vector_store import vectorstore
    from .chunker import splits
except ImportError:
    from llm_chain import model
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
    ):
        """
        Initialize hybrid retriever
        
        Args:
            documents: List of LangChain Document objects
            use_cross_encoder: Whether to use cross-encoder re-ranking
            cross_encoder_model: Model for cross-encoder
        """
        self.documents = documents
        self.doc_texts = [doc.page_content for doc in documents]
        
        # Initialize BM25 (sparse retriever)
        print("Initializing BM25 retriever...")
        tokenized_docs = [doc.split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Initialize cross-encoder if available
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE
        if self.use_cross_encoder:
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
    
    def retrieve_bm25(self, query: str, k: int = 5) -> List[Tuple[DocumentLike, float]]:
        """Retrieve using BM25 (sparse lexical search)"""
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def retrieve_dense(self, query: str, k: int = 5) -> List[Tuple[DocumentLike, float]]:
        """Retrieve using dense vectors (semantic search)"""
        docs = vectorstore.similarity_search_with_score(query, k=k)
        return docs
    
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
        bm25_results = self.retrieve_bm25(query, k=k_bm25)
        dense_results = self.retrieve_dense(query, k=k_dense)
        
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

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        results = self.hybrid_retriever.hybrid_search(
            query,
            k_bm25=self.k * 2,
            k_dense=self.k * 2,
            k_rerank=self.k,
            rerank=True,
            alpha=self.alpha,
        )
        return [doc for doc, _ in results]


class HybridQAChain:
    """QA chain using hybrid retrieval"""
    
    def __init__(self, use_hybrid: bool = True, hybrid_alpha: float = 0.5):
        """
        Initialize QA chain
        
        Args:
            use_hybrid: Whether to use hybrid retrieval (True) or just dense (False)
            hybrid_alpha: Weight for score fusion (0 = BM25, 1 = dense)
        """
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
        
        if use_hybrid:
            print("Initializing Hybrid Retriever...")
            self.hybrid_retriever = HybridRetriever(
                splits,
                use_cross_encoder=CROSS_ENCODER_AVAILABLE,
            )
            retriever = self.hybrid_retriever.as_retriever(
                k=7,
                alpha=hybrid_alpha,
            )
        else:
            print("Using Dense Retriever only...")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
        
        prompt_template = """Bạn là trợ lý tư vấn pháp luật Việt Nam.

NGUYÊN TẮC:
- Nếu câu hỏi là lời chào (ví dụ: "hello", "xin chào", "hi"...), chào lại ngắn gọn và hỏi người dùng cần tư vấn gì.
- Nếu câu hỏi liên quan pháp luật: trả lời THẲNG VÀO NỘI DUNG, không chào hỏi thêm. Trích dẫn chính xác điều khoản, mức phạt từ tài liệu.
- Nếu tài liệu có thông tin liên quan: trả lời những gì có, không nói "không đủ thông tin".
- Chỉ nói "Tài liệu không cung cấp thông tin" khi hoàn toàn không tìm thấy gì liên quan.
- Không bịa thông tin ngoài tài liệu.
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

        def format_docs(docs):
            return "\n\n".join([
                f"[Tài liệu {i+1}]:\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])

        self.format_docs = format_docs
        self.prompt = prompt
        self.model = model
        self.retriever = retriever

    def invoke(self, query_dict: Dict) -> Dict:
        """Invoke the QA chain (compatible with LangChain interface)"""
        query = query_dict.get("query", "")
        chat_history = query_dict.get("chat_history", "")

        docs = self.retriever.invoke(query)
        context = self.format_docs(docs)

        answer = (
            self.prompt
            | self.model
            | StrOutputParser()
        ).invoke({
            "chat_history": chat_history,
            "context": context,
            "question": query,
        })

        return {
            "result": answer,
            "source_documents": docs
        }
    
    def __call__(self, query: str) -> str:
        """Simple call interface"""
        result = self.invoke({"query": query})
        return result["result"]


# Export instances
print("=" * 60)
print("Initializing Hybrid Retrieval System")
print("=" * 60)

# Check if we should use hybrid retrieval (set via environment or default True)
USE_HYBRID = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() == "true"

if USE_HYBRID:
    # Create hybrid retriever
    print("\n✓ Using HYBRID retrieval (BM25 + Dense + Cross-Encoder)")
    qa_chain = HybridQAChain(use_hybrid=True, hybrid_alpha=0.5)
    hybrid_retriever = True
else:
    # Fall back to dense retrieval with custom prompt
    print("\n✓ Using DENSE retrieval only (default)")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = HybridQAChain(use_hybrid=False, hybrid_alpha=0.5)
    hybrid_retriever = False

print("=" * 60)
