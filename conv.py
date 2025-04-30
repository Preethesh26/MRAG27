# import pandas as pd
# import json

# # Load your Excel file
# df = pd.read_excel(r"C:\Users\Asus\OneDrive\Desktop\plant_formatted_final.xlsx")

# # Convert DataFrame to a list of dictionaries
# plants_data = df.to_dict(orient='records')

# # Save as JSON
# with open('D:\MRAG27\plants1.json', 'w', encoding='utf-8') as f:
#     json.dump(plants_data, f, ensure_ascii=False, indent=4)

# print("Conversion successful! plants.json created.")



# from sentence_transformers import SentenceTransformer
# import json
# import numpy as np
# import pickle

# # Load JSON data
# with open('D:/MRAG27/plants.json', 'r', encoding='utf-8') as f:
#     plants_data = json.load(f)

# # Initialize the SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Prepare the texts to embed
# texts = []
# for plant in plants_data:
#     # You can customize what to embed - here I combine plant name + uses + cures
#     combined_text = f"Plant: {plant.get('Plant Name', '')}. Uses: {plant.get('Uses', '')}. Cures: {plant.get('Cures', '')}."
#     texts.append(combined_text)

# # Generate embeddings
# embeddings = model.encode(texts)

# # Save embeddings + metadata (plant info)
# with open('D:/MRAG27/plant_embeddings.pkl', 'wb') as f:
#     pickle.dump({'embeddings': embeddings, 'plants_data': plants_data}, f)

# print("✅ Embeddings generated and saved as plant_embeddings.pkl!")

from sentence_transformers import SentenceTransformer
import chromadb
import json

# Load JSON
with open('D:/MRAG27/plants1.json', 'r', encoding='utf-8') as f:
    plants_data = json.load(f)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare texts and metadata
texts = []
metadatas = []
for idx, plant in enumerate(plants_data):
    # Create a detailed combined text for better embedding
    combined_text = (
        f"Plant Name: {plant.get('Plant Name', '')}. "
        f"Scientific Name: {plant.get('Scientific Name', '')}. "
        f"Healing Properties: {plant.get('Healing Properties', '')}. "
        f"Uses: {plant.get('Uses', '')}. "
        f"Description: {plant.get('Description', '')}. "
        f"Preparation Method: {plant.get('Preparation Method', '')}. "
        f"Side Effects: {plant.get('Side Effects', '')}. "
        f"Geographic Availability: {plant.get('Geographic Availability', '')}."
    )
    texts.append(combined_text)
    metadatas.append(plant)

# Initialize Chroma Client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or Get collection
collection = client.get_or_create_collection(name="plants_collection")

# Add documents to Chroma
collection.add(
    documents=texts,
    metadatas=metadatas,
    ids=[str(i) for i in range(len(texts))]
)

print("✅ Successfully added all plants to ChromaDB!")

