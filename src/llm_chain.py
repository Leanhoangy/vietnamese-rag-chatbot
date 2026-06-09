import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_model_name = os.getenv("EVAL_LLM_MODEL", "llama-3.3-70b-versatile")

# Load nhiều API keys để rotate khi bị rate limit
def _load_groq_keys() -> list[str]:
    keys = []
    for i in range(1, 6):
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)
    primary = os.getenv("GROQ_API_KEY", "").strip()
    if primary and primary not in keys:
        keys.insert(0, primary)
    return keys

GROQ_KEYS: list[str] = _load_groq_keys()

def make_groq_model(key_index: int = 0, model_name: str | None = None) -> ChatGroq:
    """Tạo ChatGroq với key theo index, dùng khi rotate."""
    key = GROQ_KEYS[key_index % len(GROQ_KEYS)]
    return ChatGroq(
        model=model_name or _model_name,
        api_key=key,
        temperature=0,
        max_tokens=2048,
        max_retries=1,
    )

model = make_groq_model(0)
