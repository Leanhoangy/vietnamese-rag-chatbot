from langchain_classic.chains import RetrievalQA
from llm_chain import model
from vector_store import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"verbose": False}
)

