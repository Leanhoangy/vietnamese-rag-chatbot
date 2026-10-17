import traceback

import streamlit as st

st.set_page_config(page_title="Chatbot Tư Vấn Pháp Luật", page_icon="⚖️")
st.title("⚖️ Chatbot Tư Vấn Pháp Luật Việt Nam")
st.write("Hỏi đáp về các quy định pháp luật Việt Nam dựa trên văn bản luật chính thức.")
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


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_chain():
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.environ["USE_HYBRID_RETRIEVAL"] = "true"
    from src.retrieval_hybrid import qa_chain
    return qa_chain


qa_chain = load_chain()

from src.chunker import DOC_NOTES  # noqa: E402 — sys.path đã được thêm trong load_chain()
from src.llm_chain import is_transient_error  # noqa: E402

hybrid_retriever = getattr(qa_chain, "hybrid_retriever", None)
if hybrid_retriever is not None:
    with st.sidebar:
        st.markdown("**Văn bản trong hệ thống**")
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
    message = f"{type(error).__name__}: {error}".lower()
    if is_transient_error(error):
        return ("⚠️ Hệ thống AI đang quá tải, vui lòng thử lại sau ít phút. "
                "*(The AI service is busy — please try again in a minute.)*")
    if "connect" in message:
        return "⚠️ Không kết nối được tới mô hình ngôn ngữ. Nếu chạy local bằng Ollama, hãy kiểm tra Ollama đã được bật."
    return ("⚠️ Đã có lỗi khi xử lý câu hỏi, vui lòng thử lại hoặc đặt câu hỏi theo cách khác. "
            "*(Something went wrong — please try again.)*")


examples = st.empty()
if not st.session_state.messages:
    with examples.container():
        st.markdown("**Thử một câu hỏi mẫu** · *Try a sample question (hover for English)*")
        columns = st.columns(2)
        for i, (question, english) in enumerate(EXAMPLE_QUESTIONS):
            if columns[i % 2].button(question, help=english, width="stretch", key=f"example_{i}"):
                st.session_state.pending_question = question

prompt = st.chat_input("Nhập câu hỏi về pháp luật Việt Nam...") or st.session_state.pop("pending_question", None)

if prompt:
    examples.empty()
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            try:
                chat_history = format_chat_history(st.session_state.messages[:-1])
                result = qa_chain.invoke({"query": prompt, "chat_history": chat_history})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
                search_query = result.get("search_query", "")
            except Exception as e:
                traceback.print_exc()
                answer = friendly_error(e)
                source_docs = []
                search_query = ""

        st.write(answer)

        if source_docs and len(answer) > 100:
            notes = sorted({DOC_NOTES[doc.metadata.get("law")] for doc in source_docs if doc.metadata.get("law") in DOC_NOTES})
            for note in notes:
                st.warning(note, icon="⚠️")
            with st.expander("Nguồn tài liệu"):
                if search_query and search_query != prompt:
                    st.caption(f"Truy vấn tìm kiếm: {search_query}")
                for i, doc in enumerate(source_docs, 1):
                    law = doc.metadata.get("law") or doc.metadata.get("source", "Không rõ")
                    article = doc.metadata.get("article")
                    st.markdown(f"**{i}.** {law}" + (f" — {article}" if article else ""))
                    st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
