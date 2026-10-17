from langchain_core.documents import Document

from src import observability


def test_trace_is_noop_without_keys(monkeypatch):
    monkeypatch.setattr(observability, "ENABLED", False)
    with observability.trace("chat", "question", session_id="abc", tags=["law"]) as trace:
        assert trace.config == {}
        trace.update(output="answer", metadata={"x": 1})  # không lỗi khi không có Langfuse


def test_trace_disabled_for_uploads(monkeypatch):
    monkeypatch.setattr(observability, "ENABLED", True)
    with observability.trace("chat", "question", enabled=False) as trace:
        assert trace.config == {}


def test_prepared_metadata():
    docs = [Document(page_content="a", metadata={"rerank_score": 1.23456}), Document(page_content="b")]
    metadata = observability.prepared_metadata({"result": None, "search_query": "q", "source_documents": docs})
    assert metadata == {"search_query": "q", "top_rerank_score": 1.235, "num_sources": 2, "canned_reply": False}
    refused = observability.prepared_metadata({"result": "no", "search_query": "q", "source_documents": []})
    assert refused["canned_reply"] is True and refused["top_rerank_score"] is None
