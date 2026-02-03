"""
Search Logic Module for Semantic Search.

This module provides functions to connect to the vector store and
perform similarity searches on the ingested documents.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Load environment variables
load_dotenv()

# Configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_vectors")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")

# Prompt template for LLM
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_vector_store():
    """
    Initialize and return a connection to the existing vector store.
    
    Returns:
        PGVector: The vector store instance for querying.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    
    vector_store = PGVector(
        embeddings=embeddings,
        connection=DATABASE_URL,
        collection_name=COLLECTION_NAME,
        use_jsonb=True
    )
    
    return vector_store


def search(query: str, k: int = 10) -> list:
    """
    Perform similarity search on the vector store.
    
    Args:
        query: The user's question to search for.
        k: Number of results to return (default: 10).
    
    Returns:
        List of tuples containing (document, score).
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def format_context(documents: list) -> str:
    """
    Format the retrieved documents into a context string.
    
    Args:
        documents: List of (document, score) tuples from search.
    
    Returns:
        Formatted context string with document contents.
    """
    context_parts = []
    for doc, score in documents:
        context_parts.append(doc.page_content)
    
    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Test search functionality
    print("🔍 Testing search functionality...")
    test_query = "teste"
    results = search(test_query, k=3)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Found {len(results)} results:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Result {i} (score: {score:.4f}) ---")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
        print()