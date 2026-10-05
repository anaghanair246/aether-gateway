from fastapi import FastAPI
from pydantic import BaseModel
import asyncpg

from provider import call_llm
from embedder import get_embedding
from database import init_db, DB_DSN

app = FastAPI(title="Aether Gateway")

class ChatRequest(BaseModel):
    message: str
    tier: str = "cheap"

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.post("/chat")
async def chat(request: ChatRequest):
    query_vector = get_embedding(request.message)
    
    vector_str = f"[{','.join(map(str, query_vector))}]"
    
    conn = await asyncpg.connect(DB_DSN)
    
    cache_query = """
        SELECT response, (prompt_embedding <=> $1::vector) as distance 
        FROM semantic_cache 
        ORDER BY distance ASC 
        LIMIT 1;
    """
    row = await conn.fetchrow(cache_query, vector_str)
    
    if row and row['distance'] < 0.12:
        await conn.close()
        return {
            "response": row['response'],
            "source": "cache",
            "distance_metric": round(row['distance'], 4),
            "latency_seconds": "negligible (< 50ms)"
        }
        
    llm_result = await call_llm(request.message, request.tier)
    
    insert_query = """
        INSERT INTO semantic_cache (prompt, prompt_embedding, response)
        VALUES ($1, $2::vector, $3);
    """
    await conn.execute(insert_query, request.message, vector_str, llm_result["response"])
    await conn.close()
    
    return {
        "response": llm_result["response"],
        "source": "live_llm",
        "telemetry": llm_result
    }
