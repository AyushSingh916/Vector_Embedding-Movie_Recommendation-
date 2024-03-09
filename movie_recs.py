import pymongo
import requests

client = pymongo.MongoClient("")
db = client.sample_mflix
collection = db.movies

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": "Bearer <hugging_face_token>"}

def generate_embedding(text: str) -> list[float]:
    response = requests.post(API_URL, headers=headers, json={"inputs":text})
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    return response.json()

query = "imaginary characters from outer space at war"

results = collection.aggregate([
    {"$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "plot_embedding_hf",
        "numCandidates": 100,
        "limit": 4,
        "index": "PlotSemanticSearch"
    }}
])

for document in results:
    print(f"Movie Name: {document['title']},\nMovie Plot: {document['plot']}\n")
