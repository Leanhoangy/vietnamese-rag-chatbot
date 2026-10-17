import json
from pathlib import Path

from evaluation.golden_check import check_answer

ITEM = {
    "must_include": [["18.000.000", "18 triệu"], ["20.000.000", "20 triệu"]],
    "must_cite": ["Điều 6", "168"],
    "must_not_include": ["3.000.000 đồng đến 5.000.000"],
}


def test_complete_answer_passes():
    answer = "Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng (khoản 9 Điều 6 Nghị định 168/2024/NĐ-CP)."
    assert check_answer(answer, ITEM) == []


def test_alternative_number_wording_is_accepted():
    assert check_answer("Phạt từ 18 triệu đến 20 triệu đồng theo Điều 6 Nghị định 168/2024.", ITEM) == []


def test_missing_amount_citation_and_outdated_fine_are_reported():
    answer = "Phạt tiền từ 3.000.000 đồng đến 5.000.000 đồng (Điều 5 Nghị định 100/2019/NĐ-CP)."
    problems = check_answer(answer, ITEM)
    assert "thiếu 18.000.000" in problems
    assert "không trích Điều 6" in problems
    assert any(p.startswith("có nội dung cũ") for p in problems)


def test_golden_set_file_is_well_formed():
    items = json.loads((Path(__file__).parent.parent / "evaluation" / "golden_set.json").read_text(encoding="utf-8"))
    assert len({item["id"] for item in items}) == len(items)
    for item in items:
        assert item["question"] and item["reference"] and item["domain"]
        assert all(isinstance(group, list) and group for group in item["must_include"])
