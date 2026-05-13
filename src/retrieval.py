import os

from langchain_classic.chains import RetrievalQA

try:
    from .llm_chain import model
    from .vector_store import vectorstore
except ImportError:
    from llm_chain import model
    from vector_store import vectorstore


# Default to the newer hybrid pipeline while still allowing fallback.
USE_HYBRID = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() == "true"

if USE_HYBRID:
    try:
        from .retrieval_hybrid import qa_chain
    except ImportError:
        from retrieval_hybrid import qa_chain
else:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"verbose": False},
    )
