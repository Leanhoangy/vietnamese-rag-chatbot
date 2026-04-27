
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader


loader = DirectoryLoader(
    "data", 
    glob="**/*.docx", 
    show_progress=True, 
    loader_cls=Docx2txtLoader,
    use_multithreading=True
)
data = loader.load()
