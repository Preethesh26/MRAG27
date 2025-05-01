from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
import chromadb
import json
from chromadb.utils import embedding_functions
from pathlib import Path
from fastapi.responses import JSONResponse
app = FastAPI()

# Set up template directory
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Load JSON data
with open("grouped_medicines.json", "r") as file:
    data = json.load(file)

# Initialize embedding model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-MiniLM-L6-v2")

# Setup ChromaDB client
chroma_client = chromadb.Client()
if "med_collection" in chroma_client.list_collections():
    chroma_client.delete_collection("med_collection")

collection = chroma_client.create_collection(name="med_collection", embedding_function=embedding_fn)

# Add entries to collection
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

collection.add(
    documents=[e["document"] for e in entries],
    metadatas=[e["metadata"] for e in entries],
    ids=[e["id"] for e in entries]
)

@app.get("/", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
from fastapi.responses import JSONResponse

@app.get("/api/search")
async def search_api(query: str):
    results = collection.query(query_texts=[query], n_results=5)

    if not results["metadatas"]:
        return JSONResponse(content={"message": "No matching health issue found."}, status_code=404)

    top_health_issue = results["metadatas"][0][0]["health_issue"]
    all_entries = collection.get()

    related = [
        meta for meta in all_entries["metadatas"]
        if meta["health_issue"] == top_health_issue
    ]

    return {
        "health_issue": top_health_issue,
        "results": related
    }
