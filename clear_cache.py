import asyncio
import asyncpg

DB_DSN = "postgresql://aether:password@localhost:5432/aether_db"

async def clear_cache():
    print("Connecting to database...")
    conn = await asyncpg.connect(DB_DSN)
    await conn.execute("TRUNCATE TABLE semantic_cache;")
    print("🧹 Semantic cache completely wiped!")
    await conn.close()

asyncio.run(clear_cache())
