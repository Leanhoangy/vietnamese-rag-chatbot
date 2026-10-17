import datetime
import hashlib
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Chatbot Tư Vấn Pháp Luật", page_icon="⚖️")
st.title("⚖️ Chatbot Tư Vấn Pháp Luật Việt Nam")

MODE_LAW = "⚖️ Văn bản luật · Vietnamese law"
MODE_DOCS = "📄 Tài liệu của bạn · Your documents"
mode = st.segmented_control("Chế độ · Mode", [MODE_LAW, MODE_DOCS], default=MODE_LAW, required=True,
                            key="mode", label_visibility="collapsed")

if mode == MODE_LAW:
    st.write(
        "Hỏi đáp về các quy định pháp luật Việt Nam dựa trên văn bản luật chính thức. "
        "· *Ask about Vietnamese law in Vietnamese or English.*"
    )
    st.caption(
        "⚠️ Thông tin chỉ mang tính tham khảo, được tổng hợp tự động từ các văn bản trong hệ thống "
        "và có thể chưa cập nhật quy định mới nhất. Không thay thế tư vấn của luật sư. "
        "· *For reference only — not legal advice.*"
    )
else:
    st.write(
        "Tải lên tài liệu của bạn (hợp đồng, sổ tay nhân viên, quy chế…) rồi đặt câu hỏi, "
        "hoặc dùng tài liệu mẫu. · *Upload your own documents and ask questions about them, "
        "or try the sample document.*"
    )

# Câu hỏi mẫu cho người mới (rê chuột để xem nghĩa tiếng Anh) — mỗi câu đều retrieve đúng điều luật
EXAMPLE_QUESTIONS = [
    ("Người lao động được nghỉ hằng năm bao nhiêu ngày?", "How many days of annual leave does an employee get?"),
    ("Điều kiện kết hôn theo pháp luật là gì?", "What are the legal requirements for marriage?"),
    ("Hồ sơ đăng ký thành lập công ty trách nhiệm hữu hạn gồm những gì?", "What documents are needed to register a limited liability company?"),
    ("Vượt đèn đỏ bị phạt bao nhiêu tiền?", "What is the fine for running a red light?"),
]
# Cho khách nước ngoài thử ngay: câu hỏi tiếng Anh được LLM viết lại thành truy vấn tiếng Việt
ENGLISH_EXAMPLES = [
    "How many days of annual leave does an employee get?",
    "What are the legal conditions for getting married?",
]
# Tài liệu mẫu (công ty hư cấu) cho chế độ "Tài liệu của bạn"
SAMPLE_DOC = Path(__file__).parent.parent / "samples" / "northwind_employee_handbook.md"
SAMPLE_QUESTIONS = [
    "How many days of annual leave do I get, and can I carry unused days over?",
    "What notice do I have to give if I resign?",
    "Chi phí ăn uống khi đi công tác được hỗ trợ tối đa bao nhiêu?",
]

# Demo công khai: mọi người dùng chung quota của một API key
DEMO_SESSION_LIMIT = int(os.getenv("DEMO_SESSION_LIMIT", "20"))  # câu hỏi / phiên; 0 = không giới hạn
DEMO_DAILY_LIMIT = int(os.getenv("DEMO_DAILY_LIMIT", "200"))     # câu hỏi gọi LLM / ngày cho cả app; 0 = không giới hạn
ANSWER_CACHE_TTL = 24 * 3600
ANSWER_CACHE_SIZE = 500


class DemoQuotaExceeded(Exception):
    """Demo đã dùng hết số câu hỏi gọi LLM trong ngày."""


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_chain():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.environ["USE_HYBRID_RETRIEVAL"] = "true"
    from src.retrieval_hybrid import qa_chain
    return qa_chain


qa_chain = load_chain()

from src import observability  # noqa: E402 — sys.path đã được thêm trong load_chain()
from src.canned_replies import is_small_talk  # noqa: E402
from src.chunker import DOC_NOTES  # noqa: E402
from src.document_upload import MAX_FILE_MB, MAX_FILES, SUPPORTED_SUFFIXES, UploadError  # noqa: E402
from src.llm_chain import LLM_PROVIDER, is_transient_error  # noqa: E402


@st.cache_resource
def _daily_usage() -> dict:
    """Bộ đếm dùng chung cho mọi phiên trong process."""
    return {"date": "", "count": 0, "lock": threading.Lock()}


def _take_daily_quota() -> bool:
    usage = _daily_usage()
    today = datetime.date.today().isoformat()
    with usage["lock"]:
        if usage["date"] != today:
            usage["date"], usage["count"] = today, 0
        if DEMO_DAILY_LIMIT and usage["count"] >= DEMO_DAILY_LIMIT:
            return False
        usage["count"] += 1
        return True


@st.cache_resource
def _answer_cache() -> dict:
    """Câu hỏi mở đầu giống nhau (vd. nút câu hỏi mẫu) → dùng lại câu trả lời, không tốn quota.

    Tự quản lý thay vì st.cache_data để vẫn stream được câu trả lời ở lần hỏi đầu tiên.
    Chỉ dùng cho văn bản luật (dùng chung mọi phiên), không bao giờ cho tài liệu người dùng tải lên.
    """
    return {"items": {}, "lock": threading.Lock()}


def _cache_get(question: str) -> dict | None:
    cache = _answer_cache()
    with cache["lock"]:
        item = cache["items"].get(question)
        return item[1] if item and time.time() - item[0] < ANSWER_CACHE_TTL else None


def _cache_put(question: str, reply: dict) -> None:
    cache = _answer_cache()
    with cache["lock"]:
        if len(cache["items"]) >= ANSWER_CACHE_SIZE:
            del cache["items"][min(cache["items"], key=lambda q: cache["items"][q][0])]
        cache["items"][question] = (time.time(), reply)


def format_chat_history(messages: list, max_turns: int = 5) -> str:
    """Format last N conversation turns for context."""
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(lines)


def friendly_error(error: Exception) -> str:
    """Thông báo dễ hiểu cho người dùng; lỗi chi tiết được in ra log."""
    if isinstance(error, DemoQuotaExceeded):
        return ("⚠️ Demo đã dùng hết lượt hỏi của hôm nay, vui lòng quay lại vào ngày mai. "
                "*(The demo has reached today's question limit — please come back tomorrow.)*")
    message = f"{type(error).__name__}: {error}".lower()
    if is_transient_error(error):
        return ("⚠️ Hệ thống AI đang quá tải, vui lòng thử lại sau ít phút. "
                "*(The AI service is busy — please try again in a minute.)*")
    if "connect" in message:
        return "⚠️ Không kết nối được tới mô hình ngôn ngữ. Nếu chạy local bằng Ollama, hãy kiểm tra Ollama đã được bật."
    return ("⚠️ Đã có lỗi khi xử lý câu hỏi, vui lòng thử lại hoặc đặt câu hỏi theo cách khác. "
            "*(Something went wrong — please try again.)*")


def _sources(docs: list) -> list[dict]:
    return [
        {
            "law": doc.metadata.get("law") or doc.metadata.get("source", "Không rõ"),
            "article": doc.metadata.get("article", ""),
            "text": doc.page_content,
        }
        for doc in docs
    ]


def _safe_stream(chunks, errors: list):
    """Lỗi giữa chừng (vd. 429 khi đang stream) → nối thông báo lỗi vào phần đã hiện, không làm vỡ trang."""
    streamed = False
    try:
        for chunk in chunks:
            streamed = streamed or bool(chunk)
            yield chunk
    except Exception as e:
        traceback.print_exc()
        errors.append(e)
        yield ("\n\n" if streamed else "") + friendly_error(e)


def answer_question(backend, question: str, chat_history: str, *, use_cache: bool, tags: list[str],
                    traced: bool = True) -> dict:
    """Trả lời một câu hỏi trong st.chat_message hiện tại: stream câu trả lời, trả về dict để lưu lịch sử."""
    with observability.trace("chat", question, session_id=st.session_state.session_id,
                             tags=tags, enabled=traced) as trace:
        if use_cache and (cached := _cache_get(question)):
            st.write(cached["content"])
            trace.update(output=cached["content"], metadata={"cache_hit": True})
            return dict(cached)

        try:
            if not is_small_talk(question) and not _take_daily_quota():
                raise DemoQuotaExceeded()
            with st.spinner("Đang tìm kiếm..."):
                prepared = backend.prepare(question, chat_history, config=trace.config)
        except Exception as e:
            if not isinstance(e, DemoQuotaExceeded):
                traceback.print_exc()
            st.write(content := friendly_error(e))
            trace.update(output=content, level="ERROR", status_message=type(e).__name__)
            return {"role": "assistant", "content": content}

        trace.update(metadata={**observability.prepared_metadata(prepared), "provider": LLM_PROVIDER})
        errors = []
        if prepared["result"] is not None:
            st.write(content := prepared["result"])
        else:
            content = st.write_stream(_safe_stream(backend.stream_answer(prepared, config=trace.config), errors))
            if not isinstance(content, str):
                content = "".join(str(part) for part in content)
        trace.update(output=content, **({"level": "ERROR"} if errors else {}))

        reply = {
            "role": "assistant",
            "content": content,
            "search_query": prepared.get("search_query", ""),
            "sources": _sources(prepared.get("source_documents", [])),
        }
        if use_cache and not errors:
            _cache_put(question, reply)
        return reply


def render_sources(message: dict, question: str | None) -> None:
    sources = message.get("sources") or []
    if not sources or len(message["content"]) <= 100:
        return
    for note in sorted({DOC_NOTES[source["law"]] for source in sources if source["law"] in DOC_NOTES}):
        st.warning(note, icon="⚠️")
    with st.expander("Nguồn tài liệu · Sources"):
        search_query = message.get("search_query")
        if search_query and search_query != question:
            st.caption(f"Truy vấn tìm kiếm: {search_query}")
        for i, source in enumerate(sources, 1):
            st.markdown(f"**{i}.** {source['law']}" + (f" — {source['article']}" if source["article"] else ""))
            text = source["text"]
            st.caption(text[:300] + "..." if len(text) > 300 else text)


def _docs_key(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def _clear_user_docs() -> None:
    st.session_state.doc_index = st.session_state.doc_key = st.session_state.doc_error = None
    st.session_state.doc_messages = []


def _use_sample_doc() -> None:
    st.session_state.uploader_version += 1  # xoá các tệp đang chọn trong uploader
    st.session_state.doc_request = [(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes())]


def _on_upload_change(widget_key: str) -> None:
    # Chỉ chạy khi người dùng đổi tệp. Uploader cũng rỗng sau khi chuyển chế độ (Streamlit bỏ trạng thái
    # widget không hiển thị) — trường hợp đó vẫn giữ index, nên không xoá dựa vào việc uploader rỗng
    if not st.session_state.get(widget_key):
        _clear_user_docs()


def load_user_docs() -> None:
    """Uploader + nút tài liệu mẫu; index nằm trong session_state của phiên này (không dùng chung, không cache)."""
    columns = st.columns([3, 1], vertical_alignment="bottom")
    widget_key = f"uploader_{st.session_state.uploader_version}"
    uploaded = columns[0].file_uploader(
        f"Tối đa {MAX_FILES} tệp PDF, DOCX, TXT, MD (≤ {MAX_FILE_MB} MB mỗi tệp) · "
        f"Up to {MAX_FILES} files",
        type=[suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES],
        accept_multiple_files=True,
        key=widget_key,
        on_change=_on_upload_change,
        args=(widget_key,),
    )
    columns[1].button("Dùng tài liệu mẫu · Use sample", on_click=_use_sample_doc, width="stretch")
    provider = "Gemini (Google)" if LLM_PROVIDER == "gemini" else "mô hình chạy local"
    st.caption(
        f"🔒 Tài liệu chỉ nằm trong bộ nhớ của phiên này, không được lưu lại; các đoạn liên quan tới câu hỏi "
        f"được gửi tới {provider} để trả lời. Đừng tải lên tài liệu mật. · *Files stay in this session's memory "
        f"only; passages relevant to your question are sent to the LLM. Don't upload confidential documents.*"
    )

    files = st.session_state.pop("doc_request", None)
    if files is None and uploaded:
        files = [(file.name, file.getvalue()) for file in uploaded]
    if not files:
        return
    key = _docs_key(files)
    if key == st.session_state.doc_key:
        return
    from src.user_docs import build_index

    st.session_state.doc_key = key
    st.session_state.doc_messages = []
    try:
        with st.spinner("Đang đọc và lập chỉ mục tài liệu… · Indexing your documents…"):
            st.session_state.doc_index, st.session_state.doc_error = build_index(files), None
    except UploadError as e:
        st.session_state.doc_index, st.session_state.doc_error = None, str(e)
    except Exception:
        traceback.print_exc()
        st.session_state.doc_index = None
        st.session_state.doc_error = "Không đọc được tài liệu. · Could not read the documents."


for key, default in [("messages", []), ("doc_messages", []), ("doc_index", None), ("doc_key", None),
                     ("doc_error", None), ("uploader_version", 0)]:
    st.session_state.setdefault(key, default)
st.session_state.setdefault("session_id", uuid.uuid4().hex)

hybrid_retriever = getattr(qa_chain, "hybrid_retriever", None)
if hybrid_retriever is not None:
    with st.sidebar:
        st.markdown("**Văn bản trong hệ thống** · *Documents covered*")
        laws = sorted({doc.metadata.get("law") for doc in hybrid_retriever.documents if doc.metadata.get("law")})
        st.markdown("\n".join(f"- {law}" for law in laws))

if mode == MODE_DOCS:
    load_user_docs()
    if st.session_state.doc_error:
        st.error(st.session_state.doc_error)
    elif st.session_state.doc_index is not None:
        index = st.session_state.doc_index
        st.success(f"Đang dùng · Using: {', '.join(index.file_names)} ({len(index.chunks)} đoạn · chunks)")

messages_key = "messages" if mode == MODE_LAW else "doc_messages"
messages = st.session_state[messages_key]
doc_index = st.session_state.doc_index if mode == MODE_DOCS else None

previous_question = None
for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_sources(message, previous_question)
        else:
            previous_question = message["content"]

# Hai chế độ dùng chung giới hạn câu hỏi của phiên
questions_asked = sum(1 for key in ("messages", "doc_messages")
                      for msg in st.session_state[key] if msg["role"] == "user")
session_limit_reached = bool(DEMO_SESSION_LIMIT) and questions_asked >= DEMO_SESSION_LIMIT
if session_limit_reached:
    st.info(f"Bạn đã dùng hết {DEMO_SESSION_LIMIT} câu hỏi cho phiên demo này. "
            f"*(You've reached the {DEMO_SESSION_LIMIT}-question limit for this demo session.)*")

examples = st.empty()
if not messages and (mode == MODE_LAW or doc_index is not None):
    with examples.container():
        if mode == MODE_LAW:
            st.markdown("**Thử một câu hỏi mẫu** · *Try a sample question (hover for English)*")
            columns = st.columns(2)
            for i, (question, english) in enumerate(EXAMPLE_QUESTIONS):
                if columns[i % 2].button(question, help=english, width="stretch", key=f"example_{i}"):
                    st.session_state.pending_question = question
            st.markdown("**Or ask in English**")
            columns = st.columns(2)
            for i, question in enumerate(ENGLISH_EXAMPLES):
                if columns[i % 2].button(question, width="stretch", key=f"example_en_{i}"):
                    st.session_state.pending_question = question
        elif doc_index.file_names == [SAMPLE_DOC.name]:
            st.markdown("**Thử hỏi về tài liệu mẫu** · *Try asking about the sample handbook*")
            for i, question in enumerate(SAMPLE_QUESTIONS):
                if st.button(question, width="stretch", key=f"example_doc_{i}"):
                    st.session_state.pending_question = question

waiting_for_docs = mode == MODE_DOCS and doc_index is None
prompt = st.chat_input(
    "Tải tài liệu lên trước · Upload a document first" if waiting_for_docs
    else "Nhập câu hỏi về pháp luật Việt Nam… · Ask in Vietnamese or English…" if mode == MODE_LAW
    else "Hỏi về tài liệu của bạn… · Ask about your documents…",
    disabled=session_limit_reached or waiting_for_docs,
) or st.session_state.pop("pending_question", None)

if prompt and not session_limit_reached and not waiting_for_docs:
    examples.empty()
    with st.chat_message("user"):
        st.write(prompt)
    chat_history = format_chat_history(messages)
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if mode == MODE_LAW:
            reply = answer_question(qa_chain, prompt, chat_history, use_cache=not chat_history,
                                    tags=["law", LLM_PROVIDER])
        else:
            reply = answer_question(doc_index, prompt, chat_history, use_cache=False,
                                    tags=["uploaded-docs", LLM_PROVIDER], traced=observability.TRACE_UPLOADS)
        render_sources(reply, prompt)

    messages.append(reply)
