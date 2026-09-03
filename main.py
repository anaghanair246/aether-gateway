from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI(title="Aether Gateway")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "Aether Gateway is running!"}


@app.post("/chat")
async def chat(request: ChatRequest):

    data = {
        "model": "qwen2.5:3b",
        "prompt": request.message,
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json=data
        )

    result = response.json()

    return {
        "response": result["response"]
    }