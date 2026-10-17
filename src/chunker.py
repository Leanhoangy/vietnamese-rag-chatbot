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
    "Nghidinh_168_2024.docx": "Nghị định 168/2024/NĐ-CP (xử phạt giao thông đường bộ, trừ điểm giấy phép lái xe)",
    "Nghidinh_238_2026.docx": "Nghị định 238/2026/NĐ-CP (sửa đổi, bổ sung Nghị định 168/2024)",
}

# Phần đã bị thay thế: Chương II (Điều 5–38, giao thông đường bộ) của NĐ 100/2019 — từ 01/01/2025 áp dụng
# NĐ 168/2024 (Điều 53, 54 NĐ 168). Hai văn bản có nhiều điều gần giống nhau nên NĐ 100 chiếm hết chỗ ứng viên
# khi tìm kiếm → các chunk này được gắn cờ "superseded" và chỉ được tìm khi người hỏi nhắc tới NĐ 100.
# Chương III (đường sắt) không bị NĐ 168 thay thế nên vẫn tìm bình thường.
SUPERSEDED_ARTICLES = {"Nghidinh 100:2019.docx": range(5, 39)}

# Lưu ý hiển thị kèm câu trả lời khi nguồn đã bị thay thế một phần hoặc không phải toàn văn
DOC_NOTES = {
    # Theo Điều 53, 54 Nghị định 168/2024: hiệu lực từ 01/01/2025; vi phạm trước đó áp dụng nghị định cũ
    DOC_LABELS["Nghidinh 100:2019.docx"]: (
        "Từ 01/01/2025, xử phạt vi phạm giao thông đường bộ áp dụng Nghị định 168/2024/NĐ-CP; mức phạt "
        "đường bộ của Nghị định 100/2019 chỉ áp dụng cho vi phạm xảy ra trước ngày đó. "
        "(From 1 Jan 2025 road-traffic violations are fined under Decree 168/2024; Decree 100/2019 road-traffic "
        "fines apply only to violations before that date.)"
    ),
    DOC_LABELS["Nghidinh_168_2024.docx"]: (
        "Một số điều của Nghị định 168/2024 đã được sửa đổi, bổ sung bởi Nghị định 238/2026/NĐ-CP "
        "(hiệu lực từ 15/08/2026). (Some provisions were amended by Decree 238/2026, effective 15 Aug 2026.)"
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


# Nghị định 168/2024 ghi mức trừ điểm ở khoản riêng, vd. "b) Thực hiện hành vi quy định tại điểm h khoản 5;
# khoản 6; điểm b, điểm c khoản 9 Điều này bị trừ điểm giấy phép lái xe 04 điểm" → hành vi và mức trừ điểm
# nằm ở hai chunk khác nhau, LLM không nối được. Gắn mức trừ điểm vào ngay dòng của từng hành vi.
_DEDUCTION_RULE = re.compile(
    r"^([a-zđ])\) Thực hiện hành vi quy định tại (.+?) Điều này bị trừ điểm giấy phép lái xe (\d+) điểm"
)
_CLAUSE_LINE = re.compile(r"^(\d+)\.\s")
_POINT_LINE = re.compile(r"^([a-zđ])\)\s")


def _annotate_point_deductions(article: str) -> str:
    if "bị trừ điểm giấy phép lái xe" not in article:
        return article
    lines = article.split("\n")

    deductions: dict[tuple[str, str | None], str] = {}  # (khoản, điểm hoặc None = cả khoản) → ghi chú
    clause = None
    for line in lines:
        if match := _CLAUSE_LINE.match(line):
            clause = match.group(1)
        if rule := _DEDUCTION_RULE.match(line):
            note = f"[Bị trừ {rule.group(3)} điểm giấy phép lái xe theo điểm {rule.group(1)} khoản {clause} Điều này]"
            # Đọc lần lượt: "điểm b khoản 7, điểm c khoản 9" → (7, b), (9, c); "khoản 10" không kèm điểm → cả khoản
            pending: list[str] = []
            for token in re.finditer(r"điểm ([a-zđ])\b|khoản (\d+)", rule.group(2)):
                if token.group(1):
                    pending.append(token.group(1))
                else:
                    for point in pending or [None]:
                        deductions[(token.group(2), point)] = note
                    pending = []

    annotated, clause = [], None
    for line in lines:
        if match := _CLAUSE_LINE.match(line):
            clause = match.group(1)
            if (clause, None) in deductions:
                line = f"{line} {deductions[(clause, None)]}"
        elif (match := _POINT_LINE.match(line)) and (clause, match.group(1)) in deductions:
            line = f"{line} {deductions[(clause, match.group(1))]}"
        annotated.append(line)
    return "\n".join(annotated)


def _doc_label(doc: Document) -> str:
    if doc.metadata.get("label"):  # tài liệu người dùng tải lên: luôn dùng tên tệp
        return doc.metadata["label"]
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


def _split_long_article(part: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Điều dài: gom các khoản liền nhau thành chunk ≤ max_chars, không cắt ngang khoản.

    Cắt ngang khoản làm các điểm a), b)... rời khỏi dòng mở đầu của khoản (chủ thể, mức phạt),
    LLM sẽ đọc nhầm mức phạt của khoản kế tiếp. Khoản nào vẫn quá dài thì tách tiếp,
    mỗi mảnh sau mang theo dòng mở đầu của khoản đó.
    """
    segments = [s.strip() for s in _KHOAN_PATTERN.split(part) if s.strip()]
    texts, current = [], ""
    for seg in segments:
        if len(seg) > max_chars:
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
        elif current and len(current) >= MIN_CHUNK_CHARS and len(current) + 1 + len(seg) > max_chars:
            texts.append(current)
            current = seg
        else:
            current = f"{current}\n{seg}" if current else seg
    if current:
        texts.append(current)
    return texts


def _split_by_dieu(doc: Document, max_chars: int = MAX_CHUNK_CHARS) -> list[Document]:
    """Tách document theo ranh giới Điều X., fallback RecursiveCharacter nếu điều quá dài."""
    label = _doc_label(doc)
    superseded_articles = SUPERSEDED_ARTICLES.get(Path(doc.metadata.get("source", "")).name, ())
    parts = []
    for part in (p.strip() for p in _DIEU_PATTERN.split(doc.page_content)):
        if not part:
            continue
        # Văn bản sửa đổi: "Điều 13. Sửa đổi, bổ sung Điều 42" rồi trích nguyên văn “Điều 42. …” →
        # giữ chung để chunk không rỗng và vẫn biết đây là Điều 13 của văn bản sửa đổi
        if parts and "\n" not in parts[-1] and len(parts[-1]) < MIN_CHUNK_CHARS and part.startswith(("“", '"')):
            parts[-1] = f"{parts[-1]}\n{part}"
        else:
            parts.append(part)

    chunks = []
    for part in parts:
        part = _annotate_point_deductions(part)
        first_line = part.split("\n", 1)[0].strip().lstrip('“"')
        article = _ARTICLE_PATTERN.match(first_line)
        heading = first_line[:200] if article else None
        metadata = {**doc.metadata, "law": label, "article": article.group(0) if article else ""}
        article_number = re.search(r"\d+", article.group(0)) if article else None
        metadata["superseded"] = bool(article_number and int(article_number.group(0)) in superseded_articles)

        texts = [part] if len(part) <= max_chars else _split_long_article(part, max_chars)
        chunks.extend(
            Document(page_content=_with_header(text, label, heading), metadata=dict(metadata))
            for text in texts
        )
    return chunks


def split_documents(docs: list[Document], max_chars: int = MAX_CHUNK_CHARS) -> list[Document]:
    chunks = []
    for doc in docs:
        chunks.extend(_split_by_dieu(doc, max_chars))
    return chunks


splits = split_documents(data)

# Fallback toàn bộ nếu không tìm thấy pattern Điều nào
if not splits:
    splits = _fallback_splitter.split_documents(data)

print(f"Chunked → {len(splits)} chunks")
