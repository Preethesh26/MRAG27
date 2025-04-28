from fastapi import FastAPI, Query
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="plants_collection")

# Define input schema
class QueryRequest(BaseModel):
    query: str

def search_in_chroma(query: str):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    return [
        {
            "document": doc,
            "metadata": meta
        }
        for doc, meta in zip(documents, metadatas)
    ]

# --- POST method
@app.post("/search/")
async def search_post(req: QueryRequest):
    return {"results": search_in_chroma(req.query)}

# --- GET method
@app.get("/search/")
async def search_get(query: str = Query(...)):
    return {"results": search_in_chroma(query)}
