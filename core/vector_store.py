# Vector Store module for storing and retrieving vector embeddings.

import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.documents import Document


CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

def get_vector_store(transcript : str) -> Chroma:
    # Split transcript into smaller chunks
    print(f"Building vector store from transcript...")
    print(f"Splitting transcript into chunks for vector store...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(transcript)

    # Create Document objects for each chunk
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create or load the Chroma vector store
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings_model()
    )

    # Add documents to the vector store
    vector_store.add_documents(documents)

    return vector_store

def load_vector_store() -> Chroma:
    # Load the existing Chroma vector store
    print(f"Loading existing vector store from {CHROMA_PERSIST_DIR}...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings_model()
    )
    return vector_store
 
def get_retriever(vector_store: Chroma , k: int = 5):
    # Create a retriever from the vector store
    print(f"Creating retriever from vector store with top {k} results...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5})
    return retriever