from fastapi import FastAPI
from pydantic import BaseModel
import asyncpg

# Import your existing local modules
from provider import call_llm
from embedder import get_embedding
from database import init_db, DB_DSN

app = FastAPI(title="Aether Gateway")

class ChatRequest(BaseModel):
    message: str
    tier: str = "cheap"

@app.on_event("startup")
async def startup_event():
    # Initialize the database and pgvector extension on boot
    await init_db()

@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Generate the embedding for the incoming prompt
    query_vector = get_embedding(request.message)
    
    # Format the vector for asyncpg / pgvector
    vector_str = f"[{','.join(map(str, query_vector))}]"
    
    # 2. Check the database for a semantic match
    conn = await asyncpg.connect(DB_DSN)
    
    # pgvector uses <=> for cosine distance. A distance of 0 means identical.
    # A distance of < 0.12 generally means very high semantic similarity.
    cache_query = """
        SELECT response, (prompt_embedding <=> $1::vector) as distance 
        FROM semantic_cache 
        ORDER BY distance ASC 
        LIMIT 1;
    """
    row = await conn.fetchrow(cache_query, vector_str)
    
    # 3. Cache Hit: Return instantly if similarity is high enough
    if row and row['distance'] < 0.12:
        await conn.close()
        return {
            "response": row['response'],
            "source": "cache",
            "distance_metric": round(row['distance'], 4),
            "latency_seconds": "negligible (< 50ms)"
        }
        
    # 4. Cache Miss: Route to the live LLM endpoint
    llm_result = await call_llm(request.message, request.tier)
    
    # 5. Save the new prompt, its vector, and the response to PostgreSQL
    insert_query = """
        INSERT INTO semantic_cache (prompt, prompt_embedding, response)
        VALUES ($1, $2::vector, $3);
    """
    await conn.execute(insert_query, request.message, vector_str, llm_result["response"])
    await conn.close()
    
    # 6. Return the live result
    return {
        "response": llm_result["response"],
        "source": "live_llm",
        "telemetry": llm_result
    }
