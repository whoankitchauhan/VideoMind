# Vector Store module for storing and retrieving vector embeddings.

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# -----------------------------
# Vector Store Configuration
# -----------------------------

CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "meeting_transcripts"

# Hugging Face embedding model.
# It converts text into numerical vectors.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embeddings_model():
    """
    Load the Hugging Face embedding model.

    The model converts text into vector embeddings
    so that similar pieces of text can be found later.
    """

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

    print("✓ Embedding model loaded.")

    return embeddings


def get_vector_store(transcript: str) -> Chroma:
    """
    Create a Chroma vector store from the transcript.

    Steps:
    1. Split transcript into smaller chunks.
    2. Convert chunks into Document objects.
    3. Convert chunks into embeddings.
    4. Store embeddings in ChromaDB.
    """

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty. Cannot create vector store.")

    print("\n" + "=" * 50)
    print("        Building Vector Store")
    print("=" * 50)

    # --------------------------------
    # Step 1: Split transcript
    # --------------------------------

    print("\n[1/3] Splitting transcript into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(transcript)

    print(f"✓ Transcript split into {len(chunks)} chunks.")

    # --------------------------------
    # Step 2: Create Documents
    # --------------------------------

    print("\n[2/3] Creating document objects...")

    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    print(f"✓ {len(documents)} documents created.")

    # --------------------------------
    # Step 3: Store in ChromaDB
    # --------------------------------

    print("\n[3/3] Creating ChromaDB vector store...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings_model()
    )

    vector_store.add_documents(documents)

    print("✓ Documents converted into embeddings.")
    print(f"✓ Embeddings stored in: {CHROMA_PERSIST_DIR}")

    print("\n" + "=" * 50)
    print("        Vector Store Ready")
    print("=" * 50)

    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB vector store.
    """

    print("\n" + "=" * 50)
    print("        Loading Vector Store")
    print("=" * 50)

    print(f"Database location: {CHROMA_PERSIST_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings_model()
    )

    print("✓ Vector store loaded successfully.")

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 5):
    """
    Create a retriever from the vector store.

    k = number of most relevant chunks to retrieve
        when the user asks a question.
    """

    print(f"\nCreating retriever...")
    print(f"Number of chunks to retrieve: {k}")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    print("✓ Retriever created.")

    return retriever