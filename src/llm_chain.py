import os

from dotenv import load_dotenv

load_dotenv()

# Provider sinh câu trả lời:
#   - "gemini": Google AI Studio (dùng cho Streamlit Cloud), cần GOOGLE_API_KEY
#   - "ollama": model chạy local qua Ollama, không cần key
# Không set LLM_PROVIDER → dùng gemini nếu có key, ngược lại dùng ollama.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "ollama": "qwen3:8b",
}


def _google_api_key() -> str:
    return (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()


def _resolve_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "gemini" if _google_api_key() else "ollama"
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER không hợp lệ: {provider!r} (chọn 'gemini' hoặc 'ollama')")
    return provider


LLM_PROVIDER = _resolve_provider()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[LLM_PROVIDER]


def make_llm(temperature: float = 0, max_tokens: int = 2048):
    """Tạo chat model theo LLM_PROVIDER / LLM_MODEL."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = _google_api_key()
        if not api_key:
            raise EnvironmentError("Thiếu GOOGLE_API_KEY — tạo key tại https://aistudio.google.com/apikey")
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=2,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=LLM_MODEL,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
        num_predict=max_tokens,
        num_ctx=12288,    # context mặc định của Ollama ngắn hơn prompt 7–10 chunks → bị cắt
        reasoning=False,  # tắt chế độ thinking (qwen3)
    )


def is_transient_error(error: Exception) -> bool:
    """Lỗi tạm thời của provider (quá tải / giới hạn request) — nên thử lại sau."""
    message = f"{type(error).__name__}: {error}".lower()
    return any(key in message for key in (
        "429", "503", "rate_limit", "rate limit", "resource_exhausted", "unavailable", "overloaded", "high demand",
    ))


print(f"LLM: {LLM_PROVIDER} / {LLM_MODEL}")
model = make_llm()
