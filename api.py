from fastapi import FastAPI, Form, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from pydantic import BaseModel
import json

# Shared FastAPI app instance
app = FastAPI()

# Setup CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount image directory
app.mount("/images/", StaticFiles(directory="images"), name="images")

# Templates setup
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ------------------ Model 1: Health Issue - Medicines ------------------

# Load JSON data
with open("grouped_medicines.json", "r") as file:
    data = json.load(file)

# Embedding for medicines
med_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
med_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-MiniLM-L6-v2")

# ChromaDB for medicine model
med_chroma_client = chromadb.Client()
if "med_collection" in med_chroma_client.list_collections():
    med_chroma_client.delete_collection("med_collection")

med_collection = med_chroma_client.create_collection(name="med_collection", embedding_function=med_embedding_fn)

# Add documents
entries = []
for health_issue, medicines in data.items():
    for idx, medicine in enumerate(medicines):
        entry_id = f"{health_issue}_{idx}"
        content = f"{health_issue} details: {medicine['Name of Medicine']}, Dose: {medicine['Dose and Mode of Administration']}, Indication: {medicine['Indication']}"
        entries.append({
            "id": entry_id,
            "document": content,
            "metadata": {
                "health_issue": health_issue,
                "medicine": medicine['Name of Medicine'],
                "dose": medicine['Dose and Mode of Administration'],
                "indication": medicine['Indication']
            }
        })

med_collection.add(
    documents=[e["document"] for e in entries],
    metadatas=[e["metadata"] for e in entries],
    ids=[e["id"] for e in entries]
)

@app.get("/", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/medicine_search")
async def medicine_search(query: str):
    results = med_collection.query(query_texts=[query], n_results=5)

    if not results["metadatas"]:
        return JSONResponse(content={"message": "No matching health issue found."}, status_code=404)

    top_health_issue = results["metadatas"][0][0]["health_issue"]
    all_entries = med_collection.get()

    related = [
        meta for meta in all_entries["metadatas"]
        if meta["health_issue"] == top_health_issue
    ]

    return {
        "health_issue": top_health_issue,
        "results": related
    }

# ------------------ Model 2: Plant Info ------------------

# Load second embedding model
plant_model = SentenceTransformer('all-MiniLM-L6-v2')
plant_chroma = chromadb.PersistentClient(path="./chroma_db")
plant_collection = plant_chroma.get_collection(name="plants_collection")

class QueryRequest(BaseModel):
    query: str

BASE_IMAGE_URL = "https://mrag27.onrender.com/images/"

def search_in_plants(query: str, top_k: int = 1):
    query_embedding = plant_model.encode([query]).tolist()

    results = plant_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    search_results = []
    for doc, meta in zip(documents, metadatas):
        image_filename = meta.get("Image", "")
        full_image_url = BASE_IMAGE_URL + image_filename[len("images/"):] if image_filename.startswith("images/") else BASE_IMAGE_URL + image_filename if image_filename else None

        search_results.append({
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
        })

    return search_results

@app.post("/search/")
async def plant_search_post(req: QueryRequest):
    return {"results": search_in_plants(req.query)}

@app.get("/search/")
async def plant_search_get(query: str = Query(...)):
    return {"results": search_in_plants(query)}

@app.get("/ask")
async def ask_related_plants(query: str = Query(...)):
    return {"results": search_in_plants(query, top_k=3)}

@app.get("/plant_names")
def get_plant_names():
    result = plant_collection.query(
        query_texts=["*"],
        n_results=1000
    )
    if "metadatas" not in result:
        return []

    names = list({meta["Plant Name"] for meta in result["metadatas"] if "Plant Name" in meta})
    return names
