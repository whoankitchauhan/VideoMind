from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm_utils import get_llm, invoke_llm


SECTION_KEYS = {
    "TITLE": "title",
    "SUMMARY": "summary",
    "ACTION_ITEMS": "action_items",
    "KEY_DECISIONS": "key_decisions",
    "OPEN_QUESTIONS": "open_questions",
}

EMPTY_ANALYSIS = {
    "title": "Untitled Meeting",
    "summary": "No transcript available.",
    "action_items": "No action items found.",
    "key_decisions": "No key decisions found.",
    "open_questions": "No open questions found.",
}

MAX_SINGLE_ANALYSIS_CHARS = 18000


def _analysis_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You analyze meeting/video transcripts.

Return exactly these sections and no extra commentary:

TITLE:
A professional title, maximum 8 words.

SUMMARY:
Concise bullets covering the main points and conclusions.

ACTION_ITEMS:
Numbered list with task, owner, and deadline. Use "Not specified" if missing.
If none, write "No action items found."

KEY_DECISIONS:
Numbered list of confirmed decisions only.
If none, write "No key decisions found."

OPEN_QUESTIONS:
Numbered list of unresolved questions or follow-ups.
If none, write "No open questions found."

Use only information present in the transcript."""
        ),
        ("human", "{text}")
    ])


def _parse_sections(text: str) -> dict:
    parsed = EMPTY_ANALYSIS.copy()
    current_key = None
    buffers = {value: [] for value in SECTION_KEYS.values()}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        section_name = line.rstrip(":").upper()

        if section_name in SECTION_KEYS:
            current_key = SECTION_KEYS[section_name]
            continue

        if current_key:
            buffers[current_key].append(raw_line)

    for key, lines in buffers.items():
        value = "\n".join(lines).strip()
        if value:
            parsed[key] = value

    return parsed


def _analyze_once(transcript: str, operation: str) -> dict:
    chain = _analysis_prompt() | get_llm(temperature=0.2, max_tokens=1600) | StrOutputParser()
    response = invoke_llm(
        chain,
        {"text": transcript},
        operation,
        fallback=None,
    )
    return _parse_sections(response)


def analyze_transcript(transcript: str) -> dict:
    if not transcript.strip():
        return EMPTY_ANALYSIS.copy()

    if len(transcript) <= MAX_SINGLE_ANALYSIS_CHARS:
        print("\nAnalyzing transcript in one Mistral request...")
        return _analyze_once(transcript, "analyzing the transcript")

    print("\nTranscript is long; analyzing controlled chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_SINGLE_ANALYSIS_CHARS,
        chunk_overlap=500,
    )
    chunks = splitter.split_text(transcript)

    partials = []
    for index, chunk in enumerate(chunks, start=1):
        partial = _analyze_once(chunk, f"analyzing transcript chunk {index}")
        partials.append(
            "\n".join(
                f"{label}:\n{partial[key]}"
                for label, key in SECTION_KEYS.items()
            )
        )

    combined = "\n\n---\n\n".join(partials)
    print("Combining chunk analyses in one Mistral request...")
    return _analyze_once(combined, "combining transcript analyses")
