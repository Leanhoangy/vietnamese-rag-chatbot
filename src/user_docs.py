"""
Chat với tài liệu người dùng tải lên: index tạm trong phiên (chỉ nằm trong bộ nhớ, không lưu lên đĩa),
dùng lại embedder, cross-encoder và chunker của hệ thống văn bản luật.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

try:
    from .canned_replies import is_small_talk, not_in_uploaded_docs_reply, small_talk_reply
    from .chunker import split_documents
    from .document_upload import MAX_CHUNKS, UploadError, extract_text, validate_uploads
    from .embedder import embeddings
    from .llm_chain import model
    from .retrieval_hybrid import HybridRetriever, LangChainHybridRetriever, qa_chain
except ImportError:
    from canned_replies import is_small_talk, not_in_uploaded_docs_reply, small_talk_reply
    from chunker import split_documents
    from document_upload import MAX_CHUNKS, UploadError, extract_text, validate_uploads
    from embedder import embeddings
    from llm_chain import model
    from retrieval_hybrid import HybridRetriever, LangChainHybridRetriever, qa_chain

# Prompt tiếng Anh, trung lập về ngôn ngữ: với prompt tiếng Việt, Gemini trả lời câu hỏi tiếng Anh bằng tiếng Việt
PROMPT = PromptTemplate.from_template("""You answer questions using only the DOCUMENTS the user uploaded.

RULES:
- Answer in the language of the QUESTION, whatever language the documents are in: English question → English answer; Vietnamese question → Vietnamese answer.
- Use only information from the DOCUMENTS; never make anything up. If the documents don't cover the question, say so.
- Each document excerpt starts with [file name]. Don't copy those header lines; cite briefly right after each point as (file name, section number), e.g. "(handbook.pdf, section 4)".
- Use the CHAT HISTORY to understand follow-up questions.

CHAT HISTORY:
{chat_history}

DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:""")

# Tài liệu tải lên thường ngắn và các mục ngắn → chunk nhỏ hơn văn bản luật để cross-encoder chấm chính xác hơn
CHUNK_CHARS = 400
# Cổng điểm lỏng hơn văn bản luật (-4.0): đo trên sổ tay mẫu, 10 câu có trong tài liệu (Anh + Việt) thấp nhất -4.39
# (câu tiếng Việt hỏi tài liệu tiếng Anh), còn 6 câu ngoài tài liệu từ -7.15 tới -0.77. Cross-encoder tách kém
# khi khác ngôn ngữ, nên cổng này chỉ chặn câu rõ ràng không liên quan; còn lại prompt yêu cầu LLM nói "không đề cập".
MIN_RELEVANCE_SCORE = float(os.getenv("USER_DOCS_MIN_RELEVANCE_SCORE", "-6.0"))


@dataclass
class UserDocsIndex:
    """Cùng giao diện prepare / stream_answer với HybridQAChain."""

    file_names: list[str]
    chunks: list[Document]
    retriever: LangChainHybridRetriever = field(repr=False)

    def prepare(self, question: str, chat_history: str = "", config: Dict | None = None) -> Dict:
        # Không viết lại câu hỏi: tài liệu có thể thuộc mọi lĩnh vực, từ điển thuật ngữ luật không áp dụng
        if is_small_talk(question):
            return {"result": small_talk_reply(question), "source_documents": [], "search_query": question}
        docs = self.retriever.invoke(question, config={**(config or {}), "run_name": "retrieve"})
        top_score = docs[0].metadata.get("rerank_score") if docs else None
        if not docs or (top_score is not None and top_score < MIN_RELEVANCE_SCORE):
            return {"result": not_in_uploaded_docs_reply(question), "source_documents": [], "search_query": question}
        return {
            "result": None,
            "source_documents": docs,
            "search_query": question,
            "prompt_inputs": {
                "chat_history": chat_history or "(none)",
                "context": qa_chain.format_docs(docs),
                "question": question,
            },
        }

    def stream_answer(self, prepared: Dict, config: Dict | None = None) -> Iterator[str]:
        return (PROMPT | model | StrOutputParser()).stream(
            prepared["prompt_inputs"], config={**(config or {}), "run_name": "answer"})


def build_index(files: list[tuple[str, bytes]]) -> UserDocsIndex:
    """files: (tên tệp, nội dung). Raise UploadError với thông điệp cho người dùng nếu tệp không dùng được."""
    validate_uploads([(name, len(data)) for name, data in files])
    documents = [
        Document(page_content=extract_text(name, data), metadata={"source": name, "label": name})
        for name, data in files
    ]
    chunks = split_documents(documents, max_chars=CHUNK_CHARS)
    for chunk in chunks:
        # Không có "Điều N" → dùng dòng đầu của đoạn (thường là tiêu đề mục) để hiển thị ở phần Nguồn
        if not chunk.metadata.get("article"):
            lines = chunk.page_content.split("\n", 2)
            chunk.metadata["article"] = lines[1].strip()[:60] if len(lines) > 1 else ""
    if len(chunks) > MAX_CHUNKS:
        raise UploadError(
            f"Tài liệu quá dài cho bản demo ({len(chunks)} đoạn, tối đa {MAX_CHUNKS}). "
            f"· Documents are too long for the demo ({len(chunks)} chunks, max {MAX_CHUNKS})."
        )
    store = FAISS.from_documents(chunks, embeddings)
    retriever = HybridRetriever(chunks, store=store, cross_encoder=qa_chain.hybrid_retriever.cross_encoder)
    return UserDocsIndex(
        file_names=[name for name, _ in files],
        chunks=chunks,
        retriever=LangChainHybridRetriever(hybrid_retriever=retriever, k=5),
    )
