import os
from langchain_community.vectorstores import FAISS

try:
    from .embedder import embeddings
    from .chunker import splits
except ImportError:
    from embedder import embeddings
    from chunker import splits

INDEX_PATH = "faiss_index_local"

if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("Da load FAISS index local!")
else:
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print("Da tao va luu FAISS index local!")
