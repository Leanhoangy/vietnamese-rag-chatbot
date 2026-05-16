import streamlit as st

st.set_page_config(page_title="Chatbot Tư Vấn Pháp Luật", page_icon="⚖️")
st.title("⚖️ Chatbot Tư Vấn Pháp Luật Việt Nam")
st.write("Hỏi đáp về các quy định pháp luật Việt Nam dựa trên văn bản luật chính thức.")


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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Nhập câu hỏi về pháp luật Việt Nam..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
            except Exception as e:
                answer = f"❌ Lỗi: {type(e).__name__}: {e}"
                source_docs = []

        st.write(answer)

        if source_docs:
            with st.expander("Nguồn tài liệu"):
                for i, doc in enumerate(source_docs, 1):
                    source = doc.metadata.get("source", "Không rõ")
                    st.markdown(f"**{i}.** `{source}`")
                    st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
