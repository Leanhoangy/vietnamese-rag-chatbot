import io
import zipfile

import pytest
from pypdf import PdfWriter

from src.chunker import split_documents
from src.document_upload import MAX_FILE_MB, MAX_FILES, UploadError, extract_text, validate_uploads
from langchain_core.documents import Document

SAMPLE_TEXT = "Employees receive 18 days of paid annual leave per calendar year, plus one day per 3 years of service."


def _minimal_docx(text: str) -> bytes:
    """Tệp .docx nhỏ nhất mà docx2txt đọc được: chỉ cần word/document.xml trong zip."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def test_validate_accepts_supported_files():
    validate_uploads([("handbook.pdf", 1000), ("policy.DOCX", 2000), ("notes.md", 10)])


@pytest.mark.parametrize("files", [
    [],
    [(f"f{i}.txt", 10) for i in range(MAX_FILES + 1)],
    [("slides.pptx", 10)],
    [("big.pdf", MAX_FILE_MB * 1024 * 1024 + 1)],
])
def test_validate_rejects_bad_uploads(files):
    with pytest.raises(UploadError):
        validate_uploads(files)


def test_extract_txt():
    assert extract_text("notes.txt", SAMPLE_TEXT.encode()) == SAMPLE_TEXT


def test_extract_docx():
    assert SAMPLE_TEXT in extract_text("handbook.docx", _minimal_docx(SAMPLE_TEXT))


def test_blank_pdf_is_reported_as_scanned():
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(UploadError, match="scan"):
        extract_text("scan.pdf", buffer.getvalue())


def test_too_short_text_is_rejected():
    with pytest.raises(UploadError):
        extract_text("empty.txt", b"   hi  ")


def test_uploaded_doc_label_is_file_name():
    # Nội dung có "Số: ..." (thường gặp ở hợp đồng) nhưng nhãn trích dẫn vẫn phải là tên tệp
    text = "Số: 01/2024/HĐLĐ\n1. Thời gian làm việc\n" + "Người lao động làm việc 8 giờ mỗi ngày. " * 5
    chunks = split_documents([Document(page_content=text, metadata={"source": "hopdong.pdf", "label": "hopdong.pdf"})])
    assert chunks and all(chunk.metadata["law"] == "hopdong.pdf" for chunk in chunks)
    assert chunks[0].page_content.startswith("[hopdong.pdf]")
