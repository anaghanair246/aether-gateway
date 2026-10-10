import time
import httpx
from typing import Dict, Any

MODEL_TIERS = {
    "cheap": "qwen2.5:0.5b",
    "strong": "qwen2.5:3b"
}

# Change endpoint to /api/chat to support message history
OLLAMA_URL = "http://localhost:11434/api/chat"

async def call_llm(messages: list, tier: str = "cheap") -> Dict[str, Any]:
    model_name = MODEL_TIERS.get(tier.lower(), MODEL_TIERS["cheap"])

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False
    }

    start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Error communicating with model '{model_name}': {str(e)}")

    execution_time = round(time.perf_counter() - start_time, 4)

    return {
        "tier_used": tier,
        "model_used": model_name,
        "response": data.get("message", {}).get("content", ""),
        "latency_seconds": execution_time,
        "eval_count": data.get("eval_count", 0)
    }
