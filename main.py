from fastapi import FastAPI
from pydantic import BaseModel
import asyncpg
import uuid

from provider import call_llm
from embedder import get_embedding
from database import init_db, DB_DSN
from router import router
from memory import get_chat_history, add_to_history # Import memory module

app = FastAPI(title="Aether Gateway")

class ChatRequest(BaseModel):
    message: str
    tier: str = "auto"
    session_id: str = None # Allow client to track conversations

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Session Management
    session_id = request.session_id or str(uuid.uuid4())
    
    # 2. Intelligent Routing
    if request.tier == "auto":
        route_decision = router.predict_tier(request.message)
        active_tier = route_decision["routed_tier"]
        router_telemetry = route_decision
    else:
        active_tier = request.tier
        router_telemetry = {"routed_tier": active_tier, "router_latency_ms": 0.0}

    # 3. Semantic Caching Pipeline (Only cache stand-alone queries without history)
    history = await get_chat_history(session_id)
    
    if not history: # Skip cache if in middle of a conversation
        query_vector = get_embedding(request.message)
        vector_str = f"[{','.join(map(str, query_vector))}]"
        
        conn = await asyncpg.connect(DB_DSN)
        cache_query = "SELECT response, (prompt_embedding <=> $1::vector) as distance FROM semantic_cache ORDER BY distance ASC LIMIT 1;"
        row = await conn.fetchrow(cache_query, vector_str)
        
        if row and row['distance'] < 0.12:
            await conn.close()
            # Update state with cache hit
            await add_to_history(session_id, "user", request.message)
            await add_to_history(session_id, "assistant", row['response'])
            return {
                "session_id": session_id,
                "response": row['response'],
                "source": "cache",
                "distance_metric": round(row['distance'], 4),
                "router_telemetry": router_telemetry
            }

    # 4. State Integration for Live LLM
    await add_to_history(session_id, "user", request.message)
    full_conversation = await get_chat_history(session_id)
    
    # 5. Live LLM Generation 
    llm_result = await call_llm(full_conversation, active_tier)
    
    # 6. Save LLM response to state
    await add_to_history(session_id, "assistant", llm_result["response"])
    
    # Optional: Cache the new standalone concept if it wasn't part of a conversation
    if not history:
        insert_query = "INSERT INTO semantic_cache (prompt, prompt_embedding, response) VALUES ($1, $2::vector, $3);"
        await conn.execute(insert_query, request.message, vector_str, llm_result["response"])
        await conn.close()
    
    return {
        "session_id": session_id,
        "response": llm_result["response"],
        "source": "live_llm",
        "router_telemetry": router_telemetry,
        "llm_telemetry": llm_result
    }
