import os

import requests
import tiktoken
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from groq import Groq


def setup(client):
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    documents = []
    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"}
            }
        }
    }
    index_name = "course-question"
    client.indices.create(
        index=index_name,
        body=index_settings
    )
    for doc in documents:
        client.index(index=index_name, document=doc)


def search(client, query):
    search_query = {
        "size": 3,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }

    return client.search(index="course-question", body=search_query)


def llm(question, context):
    context_template = """
    Q: {question}
    A: {text}
        """.strip()
    context_string = "\n\n".join([
        context_template.format(question=ctx["question"], text=ctx["text"]) for ctx in context
    ]).strip()

    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
    """.strip()
    prompt = prompt_template.format(
        question=question,
        context=context_string
    )
    # print(prompt)
    # print(len(prompt))

    client = Groq()
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    llm_response = completion.choices[0].message.content

    encoding = tiktoken.encoding_for_model("gpt-4o")
    print(f"Question token count: {len(encoding.encode(user_question))}")
    print(f"Response token count: {len(encoding.encode(llm_response))}")

    return llm_response


if __name__ == "__main__":
    load_dotenv()

    es_client = Elasticsearch(os.getenv("ELASTIC_SEARCH_HOST", "http://localhost:9200"))
    if not es_client.indices.exists(index="course-question"):
        setup(es_client)

    user_question = "How do copy a file to a Docker container?"
    search_result = search(es_client, user_question)
    context_list = [s["_source"] for s in search_result["hits"]["hits"]]
    # pprint.pp(context[2])

    response = llm(user_question, context_list)
    print(response)
