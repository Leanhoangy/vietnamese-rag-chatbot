import pytest

from src.canned_replies import is_small_talk, looks_like_non_answer, out_of_scope_reply, small_talk_reply


@pytest.mark.parametrize("text", [
    "hello", "Hi!", "xin chào", "Chào bạn", "chào ad ạ", "cảm ơn bạn nhé", "Thanks!", "thank you so much",
])
def test_whole_message_greetings_and_thanks_are_small_talk(text):
    assert is_small_talk(text)


@pytest.mark.parametrize("text", [
    # Có câu hỏi pháp luật kèm theo → phải đi qua pipeline bình thường
    "Chào bạn, vượt đèn đỏ bị phạt bao nhiêu tiền?",
    "Hi, how many days of annual leave does an employee get?",
    "Cảm ơn, còn người đi bộ thì sao?",
    "Điều kiện kết hôn là gì?",
])
def test_messages_with_a_question_are_not_small_talk(text):
    assert not is_small_talk(text)


def test_replies_follow_the_question_language():
    assert small_talk_reply("xin chào") == "Chào bạn! Bạn cần tư vấn về vấn đề pháp luật nào?"
    assert "Hello!" in small_talk_reply("hello")
    assert "You're welcome" in small_talk_reply("thanks")
    assert out_of_scope_reply("Cách nấu phở bò ngon?").startswith("Mình chưa tìm thấy quy định liên quan")
    assert out_of_scope_reply("How do I cook pho?").startswith("I couldn't find relevant provisions")


@pytest.mark.parametrize("answer", [
    # Câu trả lời sai thật từng bị cache trên demo
    "Tài liệu không cung cấp thông tin cụ thể về tổng số ngày nghỉ hằng năm của người lao động mà chỉ quy định ...",
    "Các tài liệu được cung cấp không đề cập đến thông tin về thủ đô của Pháp.",
    "Tài liệu không có thông tin về vấn đề này.",
    "The documents do not provide information about this.",
    "The handbook doesn't mention a notice period for interns.",
])
def test_non_answers_are_detected(answer):
    assert looks_like_non_answer(answer)


@pytest.mark.parametrize("answer", [
    "- 12 ngày làm việc đối với người làm công việc trong điều kiện bình thường (khoản 1 Điều 113 Bộ luật Lao động 2019).",
    "Người điều khiển xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt tiền từ 4.000.000 đến 6.000.000 đồng.",
    "Full-time employees receive 18 days of paid annual leave per calendar year (handbook.md, section 3).",
])
def test_real_answers_are_not_flagged(answer):
    assert not looks_like_non_answer(answer)
