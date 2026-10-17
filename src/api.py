import traceback
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from . import observability
    from .chunker import DOC_NOTES
    from .llm_chain import LLM_PROVIDER, is_transient_error
    from .retrieval_hybrid import qa_chain
except ImportError:
    import observability
    from chunker import DOC_NOTES
    from llm_chain import LLM_PROVIDER, is_transient_error
    from retrieval_hybrid import qa_chain

app = FastAPI(title="Vietnamese Legal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatTurn] = []  # các lượt trước, cũ → mới (dùng cho câu hỏi nối tiếp)


class Source(BaseModel):
    law: str
    article: str
    excerpt: str


class ChatResponse(BaseModel):
    reply: str
    search_query: str
    sources: List[Source]
    notes: List[str]


def format_history(history: List[ChatTurn], max_turns: int = 5) -> str:
    """Cùng định dạng lịch sử hội thoại với Streamlit app."""
    lines = []
    for turn in history[-(max_turns * 2):]:
        role = "Người dùng" if turn.role == "user" else "Trợ lý"
        lines.append(f"{role}: {turn.content[:200]}")
    return "\n".join(lines)


# def (không async): qa_chain chạy đồng bộ, FastAPI sẽ đưa vào threadpool thay vì chặn event loop
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        with observability.trace("api-chat", request.message, tags=["api", LLM_PROVIDER]) as trace:
            result = qa_chain.prepare(request.message, format_history(request.history), config=trace.config)
            trace.update(metadata={**observability.prepared_metadata(result), "provider": LLM_PROVIDER})
            if result["result"] is None:
                result["result"] = qa_chain.generate_answer(result, config=trace.config)
            trace.update(output=result["result"])
    except Exception as e:
        traceback.print_exc()
        if is_transient_error(e):
            raise HTTPException(status_code=503, detail="LLM provider is busy, please retry shortly.")
        raise HTTPException(status_code=500, detail="Failed to answer the question.")

    docs = result.get("source_documents", [])
    return ChatResponse(
        reply=result["result"],
        search_query=result.get("search_query", request.message),
        sources=[
            Source(
                law=doc.metadata.get("law") or doc.metadata.get("source", ""),
                article=doc.metadata.get("article", ""),
                excerpt=doc.page_content[:300],
            )
            for doc in docs
        ],
        notes=sorted({DOC_NOTES[doc.metadata.get("law")] for doc in docs if doc.metadata.get("law") in DOC_NOTES}),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
