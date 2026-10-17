"""Đọc và kiểm tra tệp người dùng tải lên (không phụ thuộc model — test được trong CI)."""

import io
from pathlib import Path

MAX_FILES = 3
MAX_FILE_MB = 10
MAX_CHUNKS = 400          # giới hạn thời gian embed trên CPU của bản demo
MIN_TEXT_CHARS = 50       # ít hơn → thường là PDF scan (chỉ có ảnh)
SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")


class UploadError(ValueError):
    """Tệp không dùng được; thông điệp hiển thị thẳng cho người dùng."""


def validate_uploads(files: list[tuple[str, int]]) -> None:
    """files: (tên tệp, kích thước byte)."""
    if not files:
        raise UploadError("Chưa có tệp nào. · No file uploaded.")
    if len(files) > MAX_FILES:
        raise UploadError(f"Tối đa {MAX_FILES} tệp mỗi lần. · Up to {MAX_FILES} files at a time.")
    for name, size in files:
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise UploadError(f"{name}: chỉ hỗ trợ PDF, DOCX, TXT, MD. · Only PDF, DOCX, TXT and MD are supported.")
        if size > MAX_FILE_MB * 1024 * 1024:
            raise UploadError(f"{name}: tệp lớn hơn {MAX_FILE_MB} MB. · File is larger than {MAX_FILE_MB} MB.")


def extract_text(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(data)).pages)
    elif suffix == ".docx":
        import docx2txt

        text = docx2txt.process(io.BytesIO(data))
    else:
        text = data.decode("utf-8", errors="replace")

    if len(text.strip()) < MIN_TEXT_CHARS:
        raise UploadError(
            f"{name}: không đọc được chữ — có thể là PDF dạng ảnh scan (bản demo chưa hỗ trợ OCR). "
            f"· No text found — possibly a scanned PDF (OCR is not enabled in the demo)."
        )
    return text
