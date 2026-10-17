"""
Giám sát bằng Langfuse (tuỳ chọn): mỗi câu hỏi là một trace gồm bước viết lại câu hỏi, tìm kiếm
(kèm các đoạn tìm được) và sinh câu trả lời (token, độ trễ), cộng metadata: provider, truy vấn
tìm kiếm, điểm rerank cao nhất, có bị từ chối hay không.

Chỉ bật khi có LANGFUSE_PUBLIC_KEY và LANGFUSE_SECRET_KEY (LANGFUSE_BASE_URL mặc định là Langfuse Cloud EU).
Không có key, hoặc chưa cài thư viện langfuse → mọi hàm ở đây là no-op.
"""

import os
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
# Chế độ "Tài liệu của bạn": mặc định không gửi câu hỏi / nội dung tài liệu lên Langfuse
TRACE_UPLOADS = os.getenv("LANGFUSE_TRACE_UPLOADS", "false").strip().lower() == "true"

if ENABLED:
    try:
        from langfuse import get_client, propagate_attributes
        from langfuse.langchain import CallbackHandler
    except ImportError:
        print("LANGFUSE_* keys are set but the langfuse package is not installed — tracing disabled.")
        ENABLED = False


class Trace:
    """Kết quả của trace(): config truyền vào LangChain, và update() để ghi metadata / output."""

    def __init__(self, span=None):
        self._span = span
        self.config = {"callbacks": [CallbackHandler()]} if span is not None else {}

    def update(self, **fields: Any) -> None:
        if self._span is not None:
            self._span.update(**fields)


@contextmanager
def trace(name: str, question: str, *, session_id: str | None = None,
          tags: list[str] | None = None, enabled: bool = True) -> Iterator[Trace]:
    if not (ENABLED and enabled):
        yield Trace()
        return
    with ExitStack() as stack:
        try:
            span = stack.enter_context(
                get_client().start_as_current_observation(name=name, as_type="chain", input=question))
            stack.enter_context(propagate_attributes(session_id=session_id, tags=tags or []))
        except Exception as e:  # lỗi của Langfuse không được làm hỏng câu trả lời
            print(f"Langfuse trace failed to start: {e}")
            span = None
        yield Trace(span)


def prepared_metadata(prepared: dict) -> dict:
    """Metadata cho trace từ kết quả prepare() của chain."""
    docs = prepared.get("source_documents") or []
    top_score = docs[0].metadata.get("rerank_score") if docs else None
    return {
        "search_query": prepared.get("search_query", ""),
        "top_rerank_score": round(float(top_score), 3) if top_score is not None else None,
        "num_sources": len(docs),
        "canned_reply": prepared.get("result") is not None,
    }
