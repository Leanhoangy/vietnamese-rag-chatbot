import os
from langchain_community.vectorstores import FAISS

try:
    from .embedder import embeddings
except ImportError:
    from embedder import embeddings


INDEX_PATH = "faiss_index_local"

if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("Đã load FAISS index local!")
else:
    try:
        from .chunker import splits
    except ImportError:
        from chunker import splits
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print("Đã tạo và lưu FAISS index local!")
