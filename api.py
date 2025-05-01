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
app.mount("/images/", StaticFiles(directory="images"), name="images")


# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_collection(name="plants_collection")

# Define input schema
class QueryRequest(BaseModel):
    query: str
BASE_IMAGE_URL = "https://mrag27.onrender.com/images/"

def search_in_chroma(query: str, top_k: int = 1):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    search_results = []
    for doc, meta in zip(documents, metadatas):
        image_filename = meta.get("Image", "")
        if image_filename:
            if image_filename.startswith("images/"):
                image_filename = image_filename[len("images/"):]
            full_image_url = BASE_IMAGE_URL + image_filename
        else:
            full_image_url = None

        plant = {
            "Plant Name": meta.get("Plant Name", ""),
            "Scientific Name": meta.get("Scientific Name", ""),
            "Healing Properties": meta.get("Healing Properties", ""),
            "Uses": meta.get("Uses", ""),
            "Description": meta.get("Description", ""),
            "Preparation Method": meta.get("Preparation Method", ""),
            "Side Effects": meta.get("Side Effects", ""),
            "Geographic Availability": meta.get("Geographic Availability", ""),
            "Image": full_image_url,
            "Image Missing": not bool(image_filename)
        }
        search_results.append(plant)

    return search_results


# --- POST method
@app.post("/search/")
async def search_post(req: QueryRequest):
    return {"results": search_in_chroma(req.query)}

# --- GET method
@app.get("/search/")
async def search_get(query: str = Query(...)):
    return {"results": search_in_chroma(query)}


@app.get("/ask")
async def ask_get(query: str = Query(...)):
    return {"results": search_in_chroma(query, top_k=3)}



@app.get("/plant_names")
def get_plant_names():
    # Use peek to get sample metadata (e.g., top 100)
    result = collection.query(
    query_texts=["*"],  # wildcard
    n_results=1000
)
 # Returns dict with 'metadatas'
    
    if "metadatas" not in result:
        return []

    # Extract unique plant names
    names = list({meta["Plant Name"] for meta in result["metadatas"] if "Plant Name" in meta})
    return names



