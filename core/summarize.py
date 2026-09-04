from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

from core.llm_utils import invoke_llm


# ------------------------------------------------------------
# Get Mistral LLM
# ------------------------------------------------------------

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


# ------------------------------------------------------------
# Split transcript into smaller chunks
# ------------------------------------------------------------

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(transcript)

    print(f"Transcript split into {len(chunks)} chunks.")

    return chunks


# ------------------------------------------------------------
# Generate meeting summary
# ------------------------------------------------------------

def summarize(transcript: str) -> str:

    if not transcript.strip():
        return "No transcript available."

    print("\nGenerating meeting summary...")

    llm = get_llm()

    # Summarize each transcript chunk
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a professional meeting summarizer.

            Summarize the given portion of the meeting transcript
            clearly and concisely.

            Focus on:
            - Important discussions
            - Decisions
            - Action items
            - Key points

            Do not add information that is not present in the transcript.
            """
        ),
        ("human", "{text}")
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")

        summary = invoke_llm(
            map_chain,
            {"text": chunk},
            f"summarizing chunk {i + 1}",
            fallback="Summary unavailable because the Mistral API rate limit was reached."
        )

        chunk_summaries.append(summary)

    # Combine all partial summaries
    combined = "\n\n".join(chunk_summaries)

    print("Combining partial summaries...")

    # Generate final summary
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting summarizer.

            Combine the provided partial summaries into one
            professional and concise meeting summary.

            Organize the final answer using bullet points.

            Include:
            - Main discussion points
            - Important decisions
            - Action items
            - Important conclusions

            Do not invent or assume information.
            Only use information present in the summaries.
            """
        ),
        ("human", "{text}")
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()

    final_summary = invoke_llm(
        combined_chain,
        {"text": combined},
        "combining summary chunks",
        fallback=combined
    )

    print("✓ Meeting summary generated.")

    return final_summary.strip()


# ------------------------------------------------------------
# Generate meeting title
# ------------------------------------------------------------

def generate_title(transcript: str) -> str:

    if not transcript.strip():
        return "Untitled Meeting"

    print("\nGenerating meeting title...")

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Based on the meeting transcript, generate a short,
            professional meeting title.

            Maximum 8 words.

            Return ONLY the title.
            Do not use quotation marks.
            """
        ),
        ("human", "{text}")
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    title = invoke_llm(
        title_chain,
        {"text": transcript[:3000]},
        "generating the meeting title",
        fallback="Untitled Meeting"
    )

    print(f"✓ Meeting title generated: {title.strip()}")

    return title.strip()
