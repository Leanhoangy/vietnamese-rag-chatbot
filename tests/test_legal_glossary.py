import pytest

from src.legal_glossary import apply_glossary, has_vietnamese_diacritics


@pytest.mark.parametrize("question, expected", [
    ("Vượt đèn đỏ bị phạt bao nhiêu tiền?", True),
    ("Điều kiện kết hôn là gì?", True),
    ("How much is the fine for running a red light?", False),
    ("vuot den do bi phat bao nhieu", False),  # tiếng Việt không dấu cũng cần LLM viết lại
    ("", False),
])
def test_detects_vietnamese_diacritics(question, expected):
    assert has_vietnamese_diacritics(question) is expected


@pytest.mark.parametrize("question, expected", [
    ("Vượt đèn đỏ bị phạt bao nhiêu tiền",
     "không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu tiền"),
    ("Đi xe máy vượt đèn vàng có bị phạt không?",
     "Đi xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông có bị phạt không?"),
    ("Không có bằng lái xe thì bị phạt bao nhiêu?", "Không có giấy phép lái xe thì bị phạt bao nhiêu?"),
    ("Uống rượu bia rồi chạy xe máy bị phạt sao?", "nồng độ cồn rồi chạy xe máy bị phạt sao?"),
    ("Công ty đuổi việc không lý do thì sao?",
     "Công ty đơn phương chấm dứt hợp đồng lao động không lý do thì sao?"),
    ("Ly dị thì ai giành quyền nuôi con?", "ly hôn thì ai quyền trực tiếp nuôi con?"),
    ("Làm sổ đỏ cần giấy tờ gì?", "Làm giấy chứng nhận quyền sử dụng đất cần giấy tờ gì?"),
])
def test_everyday_terms_become_legal_terms(question, expected):
    assert apply_glossary(question) == expected


@pytest.mark.parametrize("question", [
    # "đèn đỏ" ở đường ngang đường sắt là đúng Điều 47 NĐ 100/2019 → giữ nguyên
    "Vượt đèn đỏ khi qua đường ngang đường sắt bị phạt sao?",
    # "xe máy chuyên dùng" là thuật ngữ riêng trong luật
    "Xe máy chuyên dùng chạy trên đường cao tốc được không?",
    # "nghỉ việc riêng" là chế độ nghỉ trong Bộ luật Lao động, không phải nghỉ việc hẳn
    "Xin nghỉ việc riêng có được hưởng lương không?",
    "hello",
])
def test_protected_phrases_are_unchanged(question):
    assert apply_glossary(question) == question


def test_glossary_is_idempotent():
    once = apply_glossary("Vượt đèn đỏ và không có bằng lái thì phạt bao nhiêu?")
    assert apply_glossary(once) == once
