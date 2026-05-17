from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .loader import data
except ImportError:
    from loader import data

separators = ["\n\n", "\n", ".", " ", ""]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
    add_start_index=True,
    strip_whitespace=True,
    separators=separators,
)

splits = text_splitter.split_documents(data)
