import os
from pprint import pprint
import numpy as np

import requests
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

client = QdrantClient(os.getenv("QDRANT_HOST", "http://localhost:6333"))
# pprint(TextEmbedding.list_supported_models())

# collection_name = "zoomcamp_rag"
model_handle = "jinaai/jina-embeddings-v2-small-en"
# if not client.collection_exists(collection_name):
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
#     )
#
#     docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
#     docs_response = requests.get(docs_url)
#     documents_raw = docs_response.json()
#
#     point_id = 0
#     points = []
#     for course in documents_raw:
#         for doc in course["documents"]:
#             point = models.PointStruct(
#                 id=point_id,
#                 vector=models.Document(text=doc["text"], model=model_handle),
#                 payload={
#                     "text": doc["text"],
#                     "section": doc["section"],
#                     "course": course["course"]
#                 }
#             )
#             points.append(point)
#             point_id += 1
#
#     client.upsert(
#         collection_name=collection_name,
#         points=points
#     )

embedding_model = TextEmbedding(model_name=model_handle)
embeddings_generator = embedding_model.embed([
    "Watching porn is the worst way to release stress",
    "Maksude opo cok aku ra iso ngetest-ngetest? Ndasmu pekok!! Reneo mbahmu tak test"
])
vectors = list(embeddings_generator)
print(vectors[0].dot(vectors[1]))

# query = "I just discovered the course. Can I join now?"
# results = client.query_points(
#     collection_name=collection_name,
#     query=models.Document(
#         text=query,
#         model=model_handle
#     ),
#     limit=1,
#     with_payload=True
# )
#
# pprint(results)
