import time
import httpx
from typing import Dict, Any

MODEL_TIERS = {
    "cheap": "qwen2.5:0.5b",
    "strong": "qwen2.5:3b",
    "backup": "qwen2.5:0.5b"  # Universal failover model
}

OLLAMA_URL = "http://localhost:11434/api/chat"

async def call_llm(messages: list, tier: str = "cheap") -> Dict[str, Any]:
    primary_model = MODEL_TIERS.get(tier.lower(), MODEL_TIERS["cheap"])
    backup_model = MODEL_TIERS["backup"]

    payload = {
        "model": primary_model,
        "messages": messages,
        "stream": False
    }

    start_time = time.perf_counter()

    # Reduced timeout to 30s to trigger failover faster if the heavy model hangs
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Attempt 1: Route to the primary assigned model
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            active_model = primary_model
            breaker_status = "primary_success"
            
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"[Circuit Breaker] Primary model {primary_model} failed: {e}. Falling back to {backup_model}...")
            
            # Attempt 2: Failover to the backup model
            payload["model"] = backup_model
            try:
                response = await client.post(OLLAMA_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                active_model = backup_model
                breaker_status = "fallback_success"
            except Exception as fallback_error:
                raise RuntimeError(f"System failure: Both primary and backup models failed. {fallback_error}")

    execution_time = round(time.perf_counter() - start_time, 4)

    return {
        "tier_used": tier,
        "model_used": active_model,
        "response": data.get("message", {}).get("content", ""),
        "latency_seconds": execution_time,
        "eval_count": data.get("eval_count", 0),
        "circuit_status": breaker_status
    }
