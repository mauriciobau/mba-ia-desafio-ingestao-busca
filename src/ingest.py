"""
PDF Ingestion Script for Semantic Search.

This script loads a PDF, splits it into chunks, generates embeddings,
and stores them in PostgreSQL with pgVector.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Load environment variables
load_dotenv()

# Configuration from environment
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_vectors")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")


def ingest_pdf():
    """
    Load PDF, split into chunks, generate embeddings, and store in PostgreSQL.
    """
    print(f"📄 Loading PDF: {PDF_PATH}")
    
    # 1. Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages")
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Split into {len(chunks)} chunks")
    
    # 3. Initialize embeddings model
    print(f"🔗 Initializing embeddings model: {EMBEDDING_MODEL}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    
    # 4. Store vectors in PostgreSQL
    print(f"💾 Storing vectors in PostgreSQL (collection: {COLLECTION_NAME})")
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=DATABASE_URL,
        collection_name=COLLECTION_NAME,
        use_jsonb=True
    )
    
    print(f"✅ Ingestion complete! {len(chunks)} chunks stored successfully.")
    return vector_store


if __name__ == "__main__":
    ingest_pdf()