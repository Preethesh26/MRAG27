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
BASE_IMAGE_URL = "https://mrag27.onrender.com/images/"

def search_in_chroma(query: str):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
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
            "Plant Name": meta["Plant Name"],
            "Scientific Name": meta["Scientific Name"],
            "Healing Properties": meta["Healing Properties"],
            "Uses": meta["Uses"],
            "Description": meta["Description"],
            "Preparation Method": meta["Preparation Method"],
            "Side Effects": meta["Side Effects"],
            "Geographic Availability": meta["Geographic Availability"],
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



from transformers import pipeline

# Load T5 Question-Answering pipeline
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

@app.post("/ask")
async def ask_question(req: QueryRequest):
    user_question = req.query
    results = search_in_chroma(user_question)

    if not results:
        return {"answer": "Sorry, I couldn't find any related medicinal plant."}

    # Use the first match for context
    context = results[0]

    # Create a natural language prompt for T5
    prompt = (
        f"Context: {context['Plant Name']}, {context['Healing Properties']}, {context['Uses']}, "
        f"{context['Preparation Method']}. Question: {user_question} Answer:"
    )

    # Generate the answer
    answer = qa_model(prompt, max_length=100, do_sample=False)[0]['generated_text']

    return {"answer": answer}
