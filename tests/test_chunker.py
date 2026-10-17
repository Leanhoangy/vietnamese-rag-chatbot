import re

from langchain_core.documents import Document

from src.chunker import DOC_LABELS, DOC_NOTES, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, _split_by_dieu

DECREE = "Nghidinh 100:2019.docx"
DECREE_LABEL = DOC_LABELS[DECREE]


def make_doc(text: str, filename: str = DECREE) -> Document:
    return Document(page_content=text, metadata={"source": f"data/raw/{filename}"})


def body_lines(chunk: Document) -> list[str]:
    """Các dòng nội dung, bỏ dòng nhãn [Tên văn bản] ở đầu chunk."""
    return [line for line in chunk.page_content.split("\n")[1:] if line.strip()]


def khoan(number: int, fine: str, items: list[str]) -> str:
    lead = f"{number}. Phạt tiền {fine} đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:"
    return "\n".join([lead, *items])


def filler_items(letters: str, label: str) -> list[str]:
    return [f"{letter}) Hành vi {label} số {i} được mô tả đủ dài để làm tăng độ dài của khoản;"
            for i, letter in enumerate(letters, 1)]


def test_splits_only_on_article_headings_at_line_start():
    text = (
        "Điều 1. Phạm vi điều chỉnh\n"
        "Nghị định này áp dụng theo điểm c khoản 3 Điều 71; t) Điều 72, Điều 73.\n"
        "Điều 2. Đối tượng áp dụng\n"
        "Cá nhân, tổ chức có hành vi vi phạm."
    )
    chunks = _split_by_dieu(make_doc(text))
    assert [c.metadata["article"] for c in chunks] == ["Điều 1", "Điều 2"]


def test_every_chunk_is_prefixed_with_law_label_and_carries_metadata():
    chunks = _split_by_dieu(make_doc("Điều 5. Xử phạt người điều khiển xe ô tô\n1. Phạt tiền."))
    assert chunks[0].page_content.startswith(f"[{DECREE_LABEL}]\nĐiều 5.")
    assert chunks[0].metadata["law"] == DECREE_LABEL
    assert chunks[0].metadata["source"] == f"data/raw/{DECREE}"


def test_unknown_file_is_labelled_from_its_document_number():
    text = "Số: 168/2024/NĐ-CP\nĐiều 1. Phạm vi điều chỉnh\nNội dung."
    chunks = _split_by_dieu(make_doc(text, filename="NghiDinhMoi.docx"))
    assert all(c.metadata["law"] == "Văn bản số 168/2024/NĐ-CP" for c in chunks)


def test_quoted_heading_in_amending_law_is_recognised():
    text = "“Điều 49. Bắt buộc chữa bệnh\n1. Đối với người thực hiện hành vi nguy hiểm cho xã hội."
    chunks = _split_by_dieu(make_doc(text, filename="Luathinhsu.docx"))
    assert chunks[0].metadata["article"] == "Điều 49"


def test_amending_article_stays_with_the_quoted_article_it_introduces():
    text = (
        "Điều 13. Sửa đổi, bổ sung Điều 42\n"
        "“Điều 42. Thẩm quyền xử phạt của Chủ tịch Ủy ban nhân dân các cấp\n"
        "1. Chủ tịch Ủy ban nhân dân cấp xã có quyền phạt cảnh cáo.”\n"
        "Điều 14. Hiệu lực thi hành\n"
        "Nghị định này có hiệu lực từ ngày 15 tháng 8 năm 2026."
    )
    chunks = _split_by_dieu(make_doc(text, filename="Nghidinh_238_2026.docx"))
    assert [c.metadata["article"] for c in chunks] == ["Điều 13", "Điều 14"]
    assert "Thẩm quyền xử phạt" in chunks[0].page_content


def test_repealed_article_is_not_merged_into_the_next_one():
    chunks = _split_by_dieu(make_doc("Điều 10. (được bãi bỏ)\nĐiều 11. Phạm vi\nNội dung."))
    assert [c.metadata["article"] for c in chunks] == ["Điều 10", "Điều 11"]


def test_long_article_is_split_between_clauses_never_inside_one():
    red_light = "đ) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
    text = "\n".join([
        "Điều 8. Xử phạt người điều khiển xe đạp, xe đạp máy",
        khoan(1, "từ 80.000 đồng đến 100.000 đồng", filler_items("abcdef", "khoản 1")),
        khoan(2, "từ 100.000 đồng đến 200.000 đồng", filler_items("abcd", "khoản 2") + [red_light]),
        khoan(3, "từ 200.000 đồng đến 300.000 đồng", filler_items("abcdef", "khoản 3")),
    ])
    assert len(text) > MAX_CHUNK_CHARS
    chunks = _split_by_dieu(make_doc(text))

    assert len(chunks) >= 2
    for chunk in chunks:
        first = body_lines(chunk)[0]
        assert re.match(r"^(Điều 8\.|\d+\. )", first), f"chunk starts mid-clause: {first!r}"
    later_headers = [c.page_content.split("\n")[0] for c in chunks[1:]]
    assert all(h == f"[{DECREE_LABEL}] Điều 8. Xử phạt người điều khiển xe đạp, xe đạp máy (tiếp theo)"
               for h in later_headers)

    # Điểm đ) phải đi cùng mức phạt của khoản 2, không lẫn sang mức phạt khoản 3
    red_chunks = [c for c in chunks if red_light in c.page_content]
    assert len(red_chunks) == 1
    assert "2. Phạt tiền từ 100.000 đồng đến 200.000 đồng" in red_chunks[0].page_content
    assert "3. Phạt tiền từ 200.000 đồng" not in red_chunks[0].page_content


def test_oversized_clause_pieces_repeat_the_clause_lead():
    long_clause = khoan(2, "từ 100.000 đồng đến 200.000 đồng", filler_items("abcdefghiklmnopqrstu", "dài"))
    text = "Điều 8. Xử phạt người điều khiển xe đạp\n" + long_clause
    chunks = _split_by_dieu(make_doc(text))

    assert len(chunks) >= 2
    lead = long_clause.split("\n", 1)[0]
    for chunk in chunks[1:]:
        assert body_lines(chunk)[0] == f"{lead} (tiếp theo)"


def test_short_heading_is_never_left_as_its_own_chunk():
    # Tiêu đề + khoản 1 dài hơn MAX_CHUNK_CHARS: tiêu đề phải đi cùng khoản 1
    text = "Điều 80. Căn cứ, điều kiện thu hồi đất\n" + khoan(1, "x", filler_items("abcdefghik", "khoản 1"))
    assert len(text.split("\n", 1)[0]) < MIN_CHUNK_CHARS
    chunks = _split_by_dieu(make_doc(text))
    for chunk in chunks:
        assert len(body_lines(chunk)) > 1, f"heading-only chunk: {chunk.page_content!r}"


def test_doc_notes_refer_to_known_laws():
    assert set(DOC_NOTES) <= set(DOC_LABELS.values())
