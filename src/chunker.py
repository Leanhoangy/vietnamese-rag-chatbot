from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import data
from pprint import pprint
separators=["\n\n", "\n", ".", " ", ""]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=separators,
)

splits = text_splitter.split_documents(data)
