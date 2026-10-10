import redis.asyncio as redis
import json

# Connect to the Redis container
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

async def get_chat_history(session_id: str) -> list:
    """Retrieve the conversation history for a given session."""
    history_json = await redis_client.get(f"session:{session_id}")
    if history_json:
        return json.loads(history_json)
    return []

async def add_to_history(session_id: str, role: str, content: str):
    """Append a new message to the session's history."""
    history = await get_chat_history(session_id)
    history.append({"role": role, "content": content})
    
    # Keep only the last 10 messages to prevent context window overflow
    if len(history) > 10:
        history = history[-10:]
        
    # Save back to Redis with a 1-hour expiration
    await redis_client.setex(f"session:{session_id}", 3600, json.dumps(history))
