"""Chấm một câu trả lời theo bộ đáp án vàng (evaluation/golden_set.json). Không phụ thuộc model — chạy được trong CI."""


def check_answer(answer: str, item: dict) -> list[str]:
    """Danh sách lỗi của câu trả lời; rỗng nghĩa là đạt.

    - must_include: mỗi phần tử là một nhóm cách viết tương đương (vd. ["18.000.000", "18 triệu"]), phải có ít nhất một.
    - must_cite: các chuỗi trích dẫn bắt buộc (vd. "Điều 6", "168").
    - must_not_include: chuỗi không được xuất hiện (vd. mức phạt cũ của Nghị định 100/2019).
    """
    text = answer.lower()
    problems = []
    for alternatives in item.get("must_include", []):
        if not any(alt.lower() in text for alt in alternatives):
            problems.append(f"thiếu {alternatives[0]}")
    for citation in item.get("must_cite", []):
        if citation.lower() not in text:
            problems.append(f"không trích {citation}")
    for forbidden in item.get("must_not_include", []):
        if forbidden.lower() in text:
            problems.append(f"có nội dung cũ: {forbidden}")
    return problems
