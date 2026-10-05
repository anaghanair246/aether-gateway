from fastembed import TextEmbedding

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_embedding(text: str) -> list[float]:
    # fastembed returns a generator yielding numpy arrays
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()
