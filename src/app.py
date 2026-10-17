import datetime
import os
import threading
import traceback

import streamlit as st

st.set_page_config(page_title="Chatbot Tư Vấn Pháp Luật", page_icon="⚖️")
st.title("⚖️ Chatbot Tư Vấn Pháp Luật Việt Nam")
st.write(
    "Hỏi đáp về các quy định pháp luật Việt Nam dựa trên văn bản luật chính thức. "
    "· *Ask about Vietnamese law in Vietnamese or English.*"
)
st.caption(
    "⚠️ Thông tin chỉ mang tính tham khảo, được tổng hợp tự động từ các văn bản trong hệ thống "
    "và có thể chưa cập nhật quy định mới nhất. Không thay thế tư vấn của luật sư. "
    "· *For reference only — not legal advice.*"
)

# Câu hỏi mẫu cho người mới (rê chuột để xem nghĩa tiếng Anh) — mỗi câu đều retrieve đúng điều luật
EXAMPLE_QUESTIONS = [
    ("Người lao động được nghỉ hằng năm bao nhiêu ngày?", "How many days of annual leave does an employee get?"),
    ("Điều kiện kết hôn theo pháp luật là gì?", "What are the legal requirements for marriage?"),
    ("Hồ sơ đăng ký thành lập công ty trách nhiệm hữu hạn gồm những gì?", "What documents are needed to register a limited liability company?"),
    ("Thời hiệu khởi kiện tranh chấp hợp đồng là bao lâu?", "What is the limitation period for contract disputes?"),
]
# Cho khách nước ngoài thử ngay: câu hỏi tiếng Anh được LLM viết lại thành truy vấn tiếng Việt
ENGLISH_EXAMPLES = [
    "How many days of annual leave does an employee get?",
    "What are the legal conditions for getting married?",
]

# Demo công khai: mọi người dùng chung quota của một API key
DEMO_SESSION_LIMIT = int(os.getenv("DEMO_SESSION_LIMIT", "20"))  # câu hỏi / phiên; 0 = không giới hạn
DEMO_DAILY_LIMIT = int(os.getenv("DEMO_DAILY_LIMIT", "200"))     # câu hỏi gọi LLM / ngày cho cả app; 0 = không giới hạn


class DemoQuotaExceeded(Exception):
    """Demo đã dùng hết số câu hỏi gọi LLM trong ngày."""


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_chain():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.environ["USE_HYBRID_RETRIEVAL"] = "true"
    from src.retrieval_hybrid import qa_chain
    return qa_chain


qa_chain = load_chain()

from src.canned_replies import is_small_talk  # noqa: E402 — sys.path đã được thêm trong load_chain()
from src.chunker import DOC_NOTES  # noqa: E402
from src.llm_chain import is_transient_error  # noqa: E402


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


def run_chain(question: str, chat_history: str) -> dict:
    """Gọi QA chain (tính vào quota ngày, trừ lời chào không cần LLM), trả về dữ liệu thuần để cache được."""
    if not is_small_talk(question) and not _take_daily_quota():
        raise DemoQuotaExceeded()
    result = qa_chain.invoke({"query": question, "chat_history": chat_history})
    return {
        "answer": result["result"],
        "search_query": result.get("search_query", ""),
        "sources": [
            {
                "law": doc.metadata.get("law") or doc.metadata.get("source", "Không rõ"),
                "article": doc.metadata.get("article", ""),
                "text": doc.page_content,
            }
            for doc in result.get("source_documents", [])
        ],
    }


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def run_chain_cached(question: str) -> dict:
    """Câu hỏi mở đầu giống nhau (vd. nút câu hỏi mẫu) → dùng lại câu trả lời, không tốn quota."""
    return run_chain(question, "")


hybrid_retriever = getattr(qa_chain, "hybrid_retriever", None)
if hybrid_retriever is not None:
    with st.sidebar:
        st.markdown("**Văn bản trong hệ thống** · *Documents covered*")
        laws = sorted({doc.metadata.get("law") for doc in hybrid_retriever.documents if doc.metadata.get("law")})
        st.markdown("\n".join(f"- {law}" for law in laws))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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


questions_asked = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
session_limit_reached = bool(DEMO_SESSION_LIMIT) and questions_asked >= DEMO_SESSION_LIMIT
if session_limit_reached:
    st.info(f"Bạn đã dùng hết {DEMO_SESSION_LIMIT} câu hỏi cho phiên demo này. "
            f"*(You've reached the {DEMO_SESSION_LIMIT}-question limit for this demo session.)*")

examples = st.empty()
if not st.session_state.messages:
    with examples.container():
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

prompt = st.chat_input(
    "Nhập câu hỏi về pháp luật Việt Nam… · Ask in Vietnamese or English…",
    disabled=session_limit_reached,
) or st.session_state.pop("pending_question", None)

if prompt and not session_limit_reached:
    examples.empty()
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            try:
                chat_history = format_chat_history(st.session_state.messages[:-1])
                reply = run_chain(prompt, chat_history) if chat_history else run_chain_cached(prompt)
            except Exception as e:
                if not isinstance(e, DemoQuotaExceeded):
                    traceback.print_exc()
                reply = {"answer": friendly_error(e), "search_query": "", "sources": []}

        answer, sources = reply["answer"], reply["sources"]
        st.write(answer)

        if sources and len(answer) > 100:
            notes = sorted({DOC_NOTES[source["law"]] for source in sources if source["law"] in DOC_NOTES})
            for note in notes:
                st.warning(note, icon="⚠️")
            with st.expander("Nguồn tài liệu · Sources"):
                if reply["search_query"] and reply["search_query"] != prompt:
                    st.caption(f"Truy vấn tìm kiếm: {reply['search_query']}")
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {source['law']}" + (f" — {source['article']}" if source["article"] else ""))
                    text = source["text"]
                    st.caption(text[:300] + "..." if len(text) > 300 else text)

    st.session_state.messages.append({"role": "assistant", "content": answer})
