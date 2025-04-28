from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi.staticfiles import StaticFiles
# Initialize FastAPI app
app = FastAPI()

# CORS setup to allow cross-origin requests from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify a particular domain, like ["https://mrag27frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount /images directory
app.mount("/images", StaticFiles(directory="images"), name="images")

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
            "Plant Name": meta.get("Plant Name", "Unknown"),
            "Scientific Name": meta.get("Scientific Name", "Unknown"),
            "Healing Properties": meta.get("Healing Properties", "Not available"),
            "Uses": meta.get("Uses", "Not available"),
            "Description": doc,
            "Preparation Method": meta.get("Preparation Method", "Not available"),
            "Side Effects": meta.get("Side Effects", "Not available"),
            "Geographic Availability": meta.get("Geographic Availability", "Unknown"),
            "Image": meta.get("Image"),  # ✅ No need to modify
            "Image Missing": meta.get("Image Missing", True)
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