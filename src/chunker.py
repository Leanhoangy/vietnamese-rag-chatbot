import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .loader import data
except ImportError:
    from loader import data

# Tên văn bản gắn vào đầu mỗi chunk — theo số hiệu ghi ở đầu từng file
DOC_LABELS = {
    "Luatdansu.docx": "Bộ luật Dân sự 2015 (số 91/2015/QH13)",
    "Luatdatdai.docx": "Luật Đất đai 2024 (số 31/2024/QH15)",
    "Luatdoanhnghiep.docx": "Luật Doanh nghiệp 2020 (số 59/2020/QH14)",
    "Luatduongbo2024.docx": "Luật Đường bộ 2024 (số 35/2024/QH15)",
    "Luathinhsu.docx": "Luật sửa đổi, bổ sung Bộ luật Hình sự 2025 (số 86/2025/QH15)",
    "Luathonnhan.docx": "Luật Hôn nhân và Gia đình 2014 (số 52/2014/QH13)",
    "Luatlaodong.docx": "Bộ luật Lao động 2019 (số 45/2019/QH14)",
    "Nghidinh 100:2019.docx": "Nghị định 100/2019/NĐ-CP (xử phạt giao thông đường bộ, đường sắt)",
}

# Lưu ý hiển thị kèm câu trả lời khi nguồn đã bị thay thế một phần hoặc không phải toàn văn
DOC_NOTES = {
    DOC_LABELS["Nghidinh 100:2019.docx"]: (
        "Mức phạt giao thông đường bộ trong Nghị định 100/2019 đã được thay bằng Nghị định 168/2024/NĐ-CP "
        "từ 01/01/2025; văn bản mới chưa có trong hệ thống. "
        "(Road-traffic fines in Decree 100/2019 were replaced by Decree 168/2024 from 1 Jan 2025; "
        "the new decree is not in the system yet.)"
    ),
    DOC_LABELS["Luathinhsu.docx"]: (
        "Đây là luật sửa đổi, bổ sung một số điều của Bộ luật Hình sự, không phải toàn văn Bộ luật Hình sự. "
        "(This is the 2025 law amending the Penal Code, not the full Penal Code.)"
    ),
}

# Ranh giới điều luật: "Điều N." ở ĐẦU DÒNG (luật sửa đổi có thể mở bằng ngoặc kép).
# Neo đầu dòng để không cắt nhầm ở câu dẫn chiếu như "... khoản 3 Điều 71; t) Điều 72, Điều 73."
_DIEU_PATTERN = re.compile(r'(?=^[ \t]*[“"]?(?:Điều|ĐIỀU)\s+\d+[a-zđ]?[\.:])', re.MULTILINE)
_ARTICLE_PATTERN = re.compile(r'(?:Điều|ĐIỀU)\s+\d+[a-zđ]?')
_SO_HIEU_PATTERN = re.compile(r'(?:Bộ luật số|Luật số|Số)\s*:\s*(\S+)')
# Ranh giới khoản trong một điều: "1. ", "2. "... ở đầu dòng ("1.000.000 đồng" không khớp)
_KHOAN_PATTERN = re.compile(r'(?=^\d+\.\s)', re.MULTILINE)

MAX_CHUNK_CHARS = 1000
# Đoạn ngắn hơn mức này (thường chỉ là tiêu đề điều) không đứng thành chunk riêng
MIN_CHUNK_CHARS = 300

_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_CHARS,
    chunk_overlap=300,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)


def _doc_label(doc: Document) -> str:
    name = Path(doc.metadata.get("source", "")).name
    if name in DOC_LABELS:
        return DOC_LABELS[name]
    match = _SO_HIEU_PATTERN.search(doc.page_content[:2000])
    return f"Văn bản số {match.group(1)}" if match else Path(name).stem


def _with_header(text: str, label: str, heading: str | None) -> str:
    """Gắn tên văn bản + điều luật vào đầu chunk để LLM trích dẫn đúng và retrieval có ngữ cảnh."""
    if heading and not text.lstrip('“"').startswith(heading):
        return f"[{label}] {heading} (tiếp theo)\n{text}"
    return f"[{label}]\n{text}"


def _split_long_article(part: str) -> list[str]:
    """Điều dài: gom các khoản liền nhau thành chunk ≤ MAX_CHUNK_CHARS, không cắt ngang khoản.

    Cắt ngang khoản làm các điểm a), b)... rời khỏi dòng mở đầu của khoản (chủ thể, mức phạt),
    LLM sẽ đọc nhầm mức phạt của khoản kế tiếp. Khoản nào vẫn quá dài thì tách tiếp,
    mỗi mảnh sau mang theo dòng mở đầu của khoản đó.
    """
    segments = [s.strip() for s in _KHOAN_PATTERN.split(part) if s.strip()]
    texts, current = [], ""
    for seg in segments:
        if len(seg) > MAX_CHUNK_CHARS:
            is_khoan = re.match(r'\d+\.\s', seg) is not None
            lead = seg.split("\n", 1)[0][:300]
            if current and len(current) < MIN_CHUNK_CHARS:
                # Đoạn trước quá ngắn (thường chỉ là tiêu đề điều) → gộp vào khoản dài, tránh chunk rỗng nội dung
                seg = f"{current}\n{seg}"
            elif current:
                texts.append(current)
            current = ""
            for i, piece in enumerate(_fallback_splitter.split_text(seg)):
                texts.append(f"{lead} (tiếp theo)\n{piece}" if i > 0 and is_khoan else piece)
        elif current and len(current) >= MIN_CHUNK_CHARS and len(current) + 1 + len(seg) > MAX_CHUNK_CHARS:
            texts.append(current)
            current = seg
        else:
            current = f"{current}\n{seg}" if current else seg
    if current:
        texts.append(current)
    return texts


def _split_by_dieu(doc: Document) -> list[Document]:
    """Tách document theo ranh giới Điều X., fallback RecursiveCharacter nếu điều quá dài."""
    label = _doc_label(doc)
    parts = _DIEU_PATTERN.split(doc.page_content)
    parts = [p.strip() for p in parts if p.strip()]

    chunks = []
    for part in parts:
        first_line = part.split("\n", 1)[0].strip().lstrip('“"')
        article = _ARTICLE_PATTERN.match(first_line)
        heading = first_line[:200] if article else None
        metadata = {**doc.metadata, "law": label, "article": article.group(0) if article else ""}

        texts = [part] if len(part) <= MAX_CHUNK_CHARS else _split_long_article(part)
        chunks.extend(
            Document(page_content=_with_header(text, label, heading), metadata=dict(metadata))
            for text in texts
        )
    return chunks


def split_documents(docs: list[Document]) -> list[Document]:
    chunks = []
    for doc in docs:
        chunks.extend(_split_by_dieu(doc))
    return chunks


splits = split_documents(data)

# Fallback toàn bộ nếu không tìm thấy pattern Điều nào
if not splits:
    splits = _fallback_splitter.split_documents(data)

print(f"Chunked → {len(splits)} chunks")
