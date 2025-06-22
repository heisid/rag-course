import os
# from pprint import pprint

import requests
from qdrant_client import QdrantClient, models
# from fastembed import TextEmbedding

client = QdrantClient(os.getenv("QDRANT_HOST", "http://localhost:6333"))
# pprint(TextEmbedding.list_supported_models())

collection_name = "zoomcamp_rag"
model_handle = "jinaai/jina-embeddings-v2-small-en"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
    )

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

point_id = 0
points = []
for course in documents_raw:
    for doc in course["documents"]:
        point = models.PointStruct(
            id=point_id,
            vector=models.Document(text=doc["text"], model=model_handle),
            payload={
                "text": doc["text"],
                "section": doc["section"],
                "course": course["course"]
            }
        )
        points.append(point)
        point_id += 1

client.upsert(
    collection_name=collection_name,
    points=points
)
