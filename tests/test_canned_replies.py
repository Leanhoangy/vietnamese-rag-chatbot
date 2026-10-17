import pytest

from src.canned_replies import is_small_talk, out_of_scope_reply, small_talk_reply


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
