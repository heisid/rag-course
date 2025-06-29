import os

import requests
# from fastembed import TextEmbedding
# import numpy as np
from qdrant_client import QdrantClient, models

# Q1
# model_name = "jinaai/jina-embeddings-v2-small-en"
# query0 = "I just discovered the course. Can I join now?"
# embedding_model0 = TextEmbedding(model_name=model_name)
# embeddings_generator0 = embedding_model0.embed(query0)
# vectors_query0 = list(embeddings_generator0)
# min_val0 = min(vectors_query0[0])
# print(f"Vectorized query '{query0}' has minimum value {min_val0}")

# Q2
# doc0 = "Can I still join the course after the start date?"
# embeddings_generator1 = embedding_model0.embed([query0, doc0])
# vectors_query0_doc0 = list(embeddings_generator1)
# cosine_similarity_score0 = vectors_query0_doc0[0].dot(vectors_query0_doc0[1])
# print(f"Cosine similarity between:\nquery: '{query0}'\ndoc: '{doc0}'\nScore: {cosine_similarity_score0}")

# Q3
# doc1 = [
#     {
#         'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, "
#                 "that there will be deadlines for turning in the final projects. So don't leave everything for the "
#                 "last minute.",
#         'section': 'General course-related questions',
#         'question': 'Course - Can I still join the course after the start date?',
#         'course': 'data-engineering-zoomcamp'
#     },
#     {
#         'text': "Yes, we will keep all the materials after the course finishes, so you can follow the course at your "
#                 "own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing "
#                 "for the next cohort. I guess you can also start working on your final capstone project.",
#         'section': 'General course-related questions',
#         'question': 'Course - Can I follow the course after it finishes?',
#         'course': 'data-engineering-zoomcamp'
#     },
#     {
#         'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and "
#                 "hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office "
#                 "Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister "
#                 "before the course starts using this link.\nJoin the course Telegram channel with "
#                 "announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
#         'section': 'General course-related questions',
#         'question': 'Course - When will the course start?',
#         'course': 'data-engineering-zoomcamp'},
#     {
#         'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud '
#                 'account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the '
#                 'prerequisites and syllabus to see if you are comfortable with these subjects.',
#         'section': 'General course-related questions',
#         'question': 'Course - What can I do before the course starts?',
#         'course': 'data-engineering-zoomcamp'},
#     {
#         'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can '
#                 'improve the text or the structure of the repository.',
#         'section': 'General course-related questions',
#         'question': 'How can we contribute to the course?',
#         'course': 'data-engineering-zoomcamp'}
# ]
# doc1_text = [row['text'] for row in doc1]
# embeddings_generator2 = embedding_model0.embed(doc1_text)
# vectors_doc1 = np.array(list(embeddings_generator2))
# cosine_similarity_scores1 = vectors_doc1.dot(vectors_query0[0])
# closest_index0 = np.where(cosine_similarity_scores1 == max(cosine_similarity_scores1))[0][0]
# print(f"Query: '{query0}")
# print(f"Doc:\n", doc1_text)
# print(f"Document index with the highest similarity: {closest_index0}")

# Q4
# doc1_question_text = [f"{row['question']} {row['text']}" for row in doc1]
# embeddings_generator3 = embedding_model0.embed(doc1_question_text)
# vectors_doc1_qa = np.array(list(embeddings_generator3))
# cosine_similarity_scores2 = vectors_doc1_qa.dot(vectors_query0[0])
# closest_index1 = np.where(cosine_similarity_scores2 == max(cosine_similarity_scores2))[0][0]
# print(f"Query: '{query0}")
# print(f"Doc:\n", doc1_question_text)
# print(f"Document index with the highest similarity: {closest_index1}")

# Q5
# smallest_dim = float("inf")
# smallest_dim_model_name = ""
# for model in TextEmbedding.list_supported_models():
#     dim = model["dim"]
#     if dim < smallest_dim:
#         smallest_dim = dim
#         smallest_dim_model_name = model["model"]
# print(f"The smallest model '{smallest_dim_model_name} with dimension {smallest_dim}")


# Q6
model_handle = "BAAI/bge-small-en"
client = QdrantClient(os.getenv("QDRANT_HOST", "http://localhost:6333"))
collection_name = "vector_search"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []
    point_id = 0
    points = []
    for course in documents_raw:
        course_name = course['course']
        if course_name != 'machine-learning-zoomcamp':
            continue
        for doc in course["documents"]:
            point = models.PointStruct(
                id=point_id,
                vector=models.Document(text=f"{doc['question']} {doc['text']}", model=model_handle),
                payload={
                    "question": doc["question"],
                    "text": doc["text"],
                    "section": doc["section"],
                }
            )
            points.append(point)
            point_id += 1

    client.upsert(
        collection_name=collection_name,
        points=points
    )

query = "I just discovered the course. Can I join now?"
results = client.query_points(
    collection_name=collection_name,
    query=models.Document(
        text=query,
        model=model_handle
    ),
    limit=1,
    with_payload=True
)
first_result = results.points[0]

print(f"Query: '{query}'")
print("First result: ")
print(first_result.payload)
print(f"Similarity score: {first_result.score}")
