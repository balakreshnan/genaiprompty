from typing import List
import os
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
)
from azure.core.credentials import AzureKeyCredential

from dotenv import dotenv_values

config = dotenv_values("env.env")

azuresearchendpoint = config["AZURE_SEARCH_ENDPOINT"]
azuresearchkey = config["AZURE_SEARCH_KEY"]

def retrieve_documentation(
    question: str,
    index_name: str,
    embedding: List[float],
) -> str:

    
    search_client = SearchClient(
        endpoint=azuresearchendpoint,
        index_name=index_name,
        credential=AzureKeyCredential(azuresearchkey),
    )

    vector_query = VectorizedQuery(
        vector=embedding, k_nearest_neighbors=3, fields="contentVector"
    )

    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="default",
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=3,
    )

    docs = [
        {
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "url": doc["url"],
        }
        for doc in results
    ]

    return docs