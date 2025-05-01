import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# 1. Load the JSON data
with open("grouped_medicines.json", "r") as file:
    data = json.load(file)

# 2. Initialize SentenceTransformer
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# 3. Prepare entries and collect data per health issue
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

# 4. Setup ChromaDB and create a collection
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-MiniLM-L6-v2")

# Delete existing collection if exists
if "med_collection" in chroma_client.list_collections():
    chroma_client.delete_collection("med_collection")

collection = chroma_client.create_collection(name="med_collection", embedding_function=embedding_fn)

# 5. Add entries to ChromaDB
collection.add(
    documents=[e["document"] for e in entries],
    metadatas=[e["metadata"] for e in entries],
    ids=[e["id"] for e in entries]
)

# 6. Define search function
def query_health_issue(query: str):
    results = collection.query(query_texts=[query], n_results=1)
    if not results["metadatas"][0]:
        return []

    # Find the health issue name from the top result
    top_health_issue = results["metadatas"][0][0]["health_issue"]

    # Now search and filter all entries with the same health issue
    all_entries = collection.get()
    related_results = [
        (doc, meta)
        for doc, meta in zip(all_entries["documents"], all_entries["metadatas"])
        if meta["health_issue"] == top_health_issue
    ]
    return top_health_issue, related_results

# 7. Test it
query = "What are the medicines for abdomen?"
health_issue, results = query_health_issue(query)

print(f"Health Issue: {health_issue}")
for doc, meta in results:
    print("\nDocument:", doc)
    print("Medicine:", meta['medicine'])
    print("Dose:", meta['dose'])
    print("Indication:", meta['indication'])
