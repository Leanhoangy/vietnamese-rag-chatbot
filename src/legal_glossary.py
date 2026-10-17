"""
Quy đổi từ ngữ đời thường → thuật ngữ trong văn bản luật, áp dụng cho truy vấn trước khi retrieve.

LLM viết lại câu hỏi không ổn định với các cụm này (vd. "đèn đỏ" chỉ xuất hiện nguyên văn ở
Điều 47 NĐ 100/2019 về đường ngang đường sắt, nên BM25 kéo nhầm điều đó lên), vì vậy quy đổi
cố định bằng regex. Mọi thuật ngữ đích đều có trong data/raw/.
"""

import re

# Câu hỏi về đường sắt thì giữ nguyên "đèn đỏ" (Điều 47 NĐ 100/2019 là đúng chỗ)
_RAILWAY = re.compile(r"đường sắt|đường ngang|tàu hỏa|tàu hoả|xe lửa", re.IGNORECASE)

# (từ đời thường, thuật ngữ pháp lý, bỏ qua nếu câu khớp pattern này)
# Không quy đổi "xe máy" vì "xe máy chuyên dùng" là thuật ngữ riêng trong luật.
_GLOSSARY = [
    (r"vượt đèn (?:đỏ|vàng)", "không chấp hành hiệu lệnh của đèn tín hiệu giao thông", _RAILWAY),
    (r"đèn đỏ", "đèn tín hiệu giao thông", _RAILWAY),
    (r"bằng lái(?: xe)?", "giấy phép lái xe", None),
    (r"(?:uống |say )?rượu bia|say rượu|say xỉn", "nồng độ cồn", None),
    (r"cà vẹt(?: xe)?", "giấy đăng ký xe", None),
    (r"đuổi việc", "đơn phương chấm dứt hợp đồng lao động", None),
    (r"nghỉ ngang|(?:tự ý |tự |xin )nghỉ việc(?! riêng)", "người lao động đơn phương chấm dứt hợp đồng lao động", None),
    (r"nghỉ phép(?: năm)?", "nghỉ hằng năm", None),
    (r"tăng ca", "làm thêm giờ", None),
    (r"nghỉ đẻ|nghỉ sinh", "nghỉ thai sản", None),
    (r"ly dị", "ly hôn", None),
    (r"(?:giành |tranh )?quyền nuôi con", "quyền trực tiếp nuôi con", None),
    (r"sổ (?:đỏ|hồng)", "giấy chứng nhận quyền sử dụng đất", None),
    (r"(?:mở|lập) công ty", "thành lập doanh nghiệp", None),
    (r"vay tiền", "vay tài sản", None),
    (r"tiền cọc", "đặt cọc", None),
]

LEGAL_GLOSSARY = [
    (re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.IGNORECASE), replacement, unless)
    for pattern, replacement, unless in _GLOSSARY
]


def apply_glossary(text: str) -> str:
    """Thay từ ngữ đời thường trong câu bằng thuật ngữ pháp lý tương ứng."""
    for pattern, replacement, unless in LEGAL_GLOSSARY:
        if unless is not None and unless.search(text):
            continue
        text = pattern.sub(replacement, text)
    return text
