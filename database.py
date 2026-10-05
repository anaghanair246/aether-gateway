import asyncpg
import numpy as np

DB_DSN = "postgresql://aether:password@localhost:5432/aether_db"

async def init_db():
    conn = await asyncpg.connect(DB_DSN)
    
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create the cache table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id SERIAL PRIMARY KEY,
            prompt TEXT NOT NULL,
            prompt_embedding vector(384),
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    await conn.close()
