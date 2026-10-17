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

    # Index phải build bằng cùng embedding model dùng để query (E5: 768 chiều, custom: 128 chiều)
    _query_dim = len(embeddings.embed_query("kiểm tra"))
    if vectorstore.index.d != _query_dim:
        raise RuntimeError(
            f"FAISS index {vectorstore.index.d} chiều nhưng embedding model {_query_dim} chiều. "
            "Đặt USE_FINETUNED=true để dùng fine-tuned E5, hoặc xoá faiss_index_local/ để build lại."
        )
else:
    try:
        from .chunker import splits
    except ImportError:
        from chunker import splits
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print("Đã tạo và lưu FAISS index local!")


def add_documents(file_path: str) -> int:
    """Thêm văn bản mới vào FAISS index mà không build lại toàn bộ."""
    from langchain_community.document_loaders import Docx2txtLoader

    try:
        from .chunker import split_documents
    except ImportError:
        from chunker import split_documents

    docs = Docx2txtLoader(file_path).load()
    new_splits = split_documents(docs)  # cùng cách chunk + gắn tên văn bản như lúc build index
    vectorstore.add_documents(new_splits)
    vectorstore.save_local(INDEX_PATH)
    print(f"Đã thêm {len(new_splits)} chunks từ {file_path}")
    return len(new_splits)
