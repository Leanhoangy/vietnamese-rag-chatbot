import requests
import streamlit as st
st.set_page_config(page_title="Chatbot Tư Vấn Pháp Luật", page_icon="⚖️")
st.title("⚖️ Chatbot Tư Vấn Pháp Luật Việt Nam")
st.write("Hỏi đáp về các quy định pháp luật Việt Nam dựa trên văn bản luật chính thức.")

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
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"message": prompt},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()["reply"]
            except requests.exceptions.ConnectionError:
                result = "❌ Không kết nối được API. Hãy chạy: `uvicorn src.api:app --reload`"
            except requests.exceptions.Timeout:
                result = "❌ API timeout. Thử lại sau."
            except Exception as e:
                result = f"❌ Lỗi: {e}"
        st.write(result)
    st.session_state.messages.append({"role": "assistant", "content": result})

