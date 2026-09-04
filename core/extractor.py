from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

from core.llm_utils import invoke_llm


# ------------------------------------------------------------
# Get Mistral LLM
# ------------------------------------------------------------

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2
    )


# ------------------------------------------------------------
# Build reusable analysis chain
# ------------------------------------------------------------

def build_chain(system_prompt: str):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    return prompt | llm | StrOutputParser()


# ------------------------------------------------------------
# Extract Action Items
# ------------------------------------------------------------

def extract_action_items(transcript: str) -> str:
    if not transcript.strip():
        return "No action items found."

    print("\nExtracting action items...")

    chain = build_chain(
        """
        You are an expert meeting analyst.

        From the meeting transcript, extract all action items.

        For each action item provide:
        - Task description
        - Owner or responsible person
        - Deadline, if mentioned. Otherwise write "Not specified".

        Only use information explicitly present in the transcript.
        Do not guess or invent owners or deadlines.

        Format the result as a numbered list.

        If no action items are found, say:
        "No action items found."
        """
    )

    result = invoke_llm(
        chain,
        {"text": transcript},
        "extracting action items",
        fallback="Action items unavailable because the Mistral API rate limit was reached."
    )

    print("✓ Action items extracted.")

    return result.strip()


# ------------------------------------------------------------
# Extract Key Decisions
# ------------------------------------------------------------

def extract_key_decisions(transcript: str) -> str:
    if not transcript.strip():
        return "No key decisions found."

    print("\nExtracting key decisions...")

    chain = build_chain(
        """
        You are an expert meeting analyst.

        From the meeting transcript, extract all important
        decisions that were actually made.

        Only include confirmed decisions.
        Do not include suggestions, possibilities, or discussions
        that did not result in a decision.

        Format the result as a numbered list.

        If no key decisions are found, say:
        "No key decisions found."
        """
    )

    result = invoke_llm(
        chain,
        {"text": transcript},
        "extracting key decisions",
        fallback="Key decisions unavailable because the Mistral API rate limit was reached."
    )

    print("✓ Key decisions extracted.")

    return result.strip()


# ------------------------------------------------------------
# Extract Open Questions
# ------------------------------------------------------------

def extract_questions(transcript: str) -> str:
    if not transcript.strip():
        return "No open questions found."

    print("\nExtracting open questions...")

    chain = build_chain(
        """
        You are an expert meeting analyst.

        From the meeting transcript, identify all unresolved
        questions, unclear points, or topics requiring follow-up.

        Only include questions or issues that are actually
        unresolved in the transcript.

        Do not invent questions.

        Format the result as a numbered list.

        If no open questions are found, say:
        "No open questions found."
        """
    )

    result = invoke_llm(
        chain,
        {"text": transcript},
        "extracting open questions",
        fallback="Open questions unavailable because the Mistral API rate limit was reached."
    )

    print("✓ Open questions extracted.")

    return result.strip()
