from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .retrieval import qa_chain
except ImportError:
    from retrieval import qa_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = qa_chain.invoke({"query": request.message})
    return {"reply": response['result']}

@app.get("/health")
async def health():    
    return {"status": "ok"}
