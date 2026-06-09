import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_model_name = os.getenv("EVAL_LLM_MODEL", "llama-3.3-70b-versatile")

model = ChatGroq(
    model=_model_name,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=2048,
    max_retries=2,
)
