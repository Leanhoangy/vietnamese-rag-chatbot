"""
Câu trả lời cố định, không cần gọi LLM: lời chào / cảm ơn, và câu hỏi ngoài phạm vi văn bản của hệ thống.
Ngôn ngữ trả lời theo câu hỏi (có dấu tiếng Việt → tiếng Việt; còn lại → song ngữ hoặc tiếng Anh).
"""

import re

try:
    from .legal_glossary import has_vietnamese_diacritics
except ImportError:
    from legal_glossary import has_vietnamese_diacritics

# Chỉ khớp khi CẢ câu là lời chào/cảm ơn — "chào bạn, vượt đèn đỏ phạt bao nhiêu?" vẫn đi qua pipeline
_SUFFIX = r"(?:\s+(?:bạn|nhé|nha|ạ|ad|admin|there|so much))*[\s!.,?~]*"
_GREETING = re.compile(
    r"^\s*(?:xin chào|chào|hello|hi|hey|alo|good (?:morning|afternoon|evening))" + _SUFFIX + r"$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^\s*(?:cảm ơn|cám ơn|thanks|thank you|thank)" + _SUFFIX + r"$", re.IGNORECASE)


def is_small_talk(text: str) -> bool:
    return bool(_GREETING.match(text) or _THANKS.match(text))


def small_talk_reply(text: str) -> str:
    vietnamese = has_vietnamese_diacritics(text)
    if _THANKS.match(text):
        if vietnamese:
            return "Rất vui được hỗ trợ bạn! Bạn còn câu hỏi pháp luật nào khác không?"
        return "You're welcome! · Rất vui được hỗ trợ bạn! Any other legal question?"
    if vietnamese:
        return "Chào bạn! Bạn cần tư vấn về vấn đề pháp luật nào?"
    return "Xin chào! Bạn cần tư vấn về vấn đề pháp luật nào? · Hello! What legal question can I help you with?"


def out_of_scope_reply(question: str) -> str:
    if has_vietnamese_diacritics(question):
        return (
            "Mình chưa tìm thấy quy định liên quan trong các văn bản của hệ thống. Hiện hệ thống hỗ trợ: "
            "Bộ luật Dân sự, Bộ luật Lao động, Luật Doanh nghiệp, Luật Đất đai, Luật Hôn nhân và Gia đình, "
            "Luật Đường bộ, các Nghị định 168/2024, 238/2026 và 100/2019 về xử phạt giao thông "
            "và luật sửa đổi Bộ luật Hình sự 2025. "
            "Bạn thử hỏi cụ thể hơn hoặc dùng thuật ngữ pháp lý nhé."
        )
    return (
        "I couldn't find relevant provisions in the documents this system covers: the Civil Code, Labor Code, "
        "Law on Enterprises, Land Law, Law on Marriage and Family, Road Law, Decrees 168/2024, 238/2026 "
        "and 100/2019 on traffic fines "
        "and the 2025 amendments to the Penal Code. Try asking more specifically."
    )
