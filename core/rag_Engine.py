import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    get_vector_store,
    load_vector_store,
    get_retriever
)


# --------------------------------
# Mistral LLM
# --------------------------------

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


# --------------------------------
# Format retrieved documents
# --------------------------------

def format_docs(docs):
    """
    Convert retrieved Document objects
    into one normal text string.
    """

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------
# RAG Chain - New Transcript
# --------------------------------

def build_rag_chain(transcript: str):

    print("\n" + "=" * 50)
    print("        Building RAG Chain")
    print("=" * 50)

    # Step 1: Create vector store
    print("\n[1/3] Creating vector store...")
    vector_store = get_vector_store(transcript)

    # Step 2: Create retriever
    print("\n[2/3] Creating retriever...")
    retriever = get_retriever(vector_store, k=4)

    # Step 3: Load LLM
    print("\n[3/3] Loading Mistral LLM...")
    llm = get_llm()

    # --------------------------------
    # Prompt
    # --------------------------------

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise.

If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    # --------------------------------
    # LCEL RAG Pipeline
    # --------------------------------

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✓ RAG chain created successfully.")

    return rag_chain


# --------------------------------
# RAG Chain - Existing Database
# --------------------------------

def load_rag_chain():

    print("\n" + "=" * 50)
    print("        Loading Existing RAG Chain")
    print("=" * 50)

    # Load existing ChromaDB
    print("\n[1/3] Loading vector store...")
    vector_store = load_vector_store()

    # Create retriever
    print("\n[2/3] Creating retriever...")
    retriever = get_retriever(vector_store, k=4)

    # Load Mistral
    print("\n[3/3] Loading Mistral LLM...")
    llm = get_llm()

    # Same prompt as above
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise.

If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    # RAG pipeline
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✓ Existing RAG chain loaded successfully.")

    return rag_chain


# --------------------------------
# Ask Question
# --------------------------------

def ask_question(rag_chain, question: str) -> str:

    print("\n" + "-" * 50)
    print(f"Question: {question}")
    print("-" * 50)

    answer = rag_chain.invoke(question)

    print(f"\nAnswer:\n{answer}")

    return answer