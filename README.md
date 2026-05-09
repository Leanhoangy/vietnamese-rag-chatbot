# ⚖️ Vietnamese RAG Chatbot — Tư Vấn Pháp Luật

> Chatbot hỏi đáp pháp luật Việt Nam sử dụng Retrieval-Augmented Generation (RAG)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)](https://vietnamese-rag-chatbot-jemrcpipbmxd5bdugx5z2l.streamlit.app)

## 🔗 Live Demo
👉 [vietnamese-rag-chatbot-jemrcpipbmxd5bdugx5z2l.streamlit.app](https://vietnamese-rag-chatbot-jemrcpipbmxd5bdugx5z2l.streamlit.app)

---

## 📌 Giới thiệu

Hệ thống chatbot cho phép người dùng **hỏi đáp về pháp luật Việt Nam** dựa trên các văn bản luật chính thức (Luật Đường bộ, Bộ luật Lao động, Luật Doanh nghiệp...).

## 🏗️ Kiến trúc RAG Pipeline

```
Tài liệu .docx
      ↓
[Chunking] RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
      ↓
[Embedding] multilingual-e5-base (HuggingFace)
      ↓
[Vector Store] FAISS
      
── Khi user hỏi ──

Câu hỏi → Embed → FAISS top-k → Prompt → Groq (Llama 3.1) → Câu trả lời
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain 0.3 |
| Embedding | multilingual-e5-base (HuggingFace) |
| Vector Store | FAISS |
| LLM | Llama 3.1-8b (Groq API) |
| UI | Streamlit |
| Evaluation | RAGAS |

## 📊 Kết quả RAGAS Evaluation

| Metric | Score |
|--------|-------|
| Faithfulness | 0.867 |
| Answer Relevancy | 1.000 |
| Context Precision | 0.933 |
| Context Recall | 0.800 |

## 🚀 Cài đặt và chạy

```bash
# Clone repo
git clone https://github.com/Leanhoangy/vietnamese-rag-chatbot.git
cd vietnamese-rag-chatbot

# Cài thư viện
pip install -r requirements.txt

# Tạo file .env
cp .env.example .env
# Điền GROQ_API_KEY vào .env

# Chạy app
streamlit run src/app.py
```

## 📁 Cấu trúc project

```
vietnamese-rag-chatbot/
├── data/
│   └── raw/          # Văn bản luật .docx
├── src/
│   ├── loader.py      # Đọc file .docx
│   ├── chunker.py     # Chia text thành chunks
│   ├── embedder.py    # Load embedding model
│   ├── vector_store.py # Build/load FAISS index
│   ├── llm_chain.py   # Kết nối Groq API
│   ├── retrieval.py   # RAG pipeline
│   ├── evaluation.py  # RAGAS evaluation
│   └── app.py         # Streamlit UI
├── test_set.json      # 10 câu hỏi test
├── ragas_results.csv  # Kết quả evaluation
└── requirements.txt
```

## 👥 Nhóm thực hiện

Đồ án môn Deep Learning — HUIT 2025

