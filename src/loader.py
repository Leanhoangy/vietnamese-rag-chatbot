
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader


docx_loader = DirectoryLoader(
    "data",
    glob="**/*.docx",
    show_progress=True,
    loader_cls=Docx2txtLoader,
    use_multithreading=True,
)

doc_loader = DirectoryLoader(
    "data",
    glob="**/*.doc",
    show_progress=True,
    loader_cls=UnstructuredWordDocumentLoader,
    use_multithreading=True,
)

pdf_loader = DirectoryLoader(
    "data",
    glob="**/*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader,
    use_multithreading=True,
)

data = docx_loader.load() + doc_loader.load() + pdf_loader.load()
print(f"Loaded {len(data)} documents total")
