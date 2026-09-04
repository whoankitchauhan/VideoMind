from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)


from core.rag_Engine import build_rag_chain, ask_question


# Load API keys and environment variables from .env
load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:

    print("\n" + "=" * 60)
    print("           Starting VideoMind")
    print("=" * 60)

    # --------------------------------
    # STEP 1: Audio Processing
    # --------------------------------

    print("\n[1/7] Processing video/audio...")

    chunks = process_input(source)

    # --------------------------------
    # STEP 2: Transcription
    # --------------------------------

    print("\n[2/7] Transcribing audio...")

    transcript = transcribe_audio_chunks(chunks, language)

    print(
        f"\nRaw transcription "
        f"(first 300 characters):\n{transcript[:300]}"
    )

    # --------------------------------
    # STEP 3: Generate Title
    # --------------------------------

    print("\n[3/7] Generating title...")

    title = generate_title(transcript)

    # --------------------------------
    # STEP 4: Generate Summary
    # --------------------------------

    print("\n[4/7] Generating summary...")

    summary = summarize(transcript)

    # --------------------------------
    # STEP 5: Extract Information
    # --------------------------------

    print("\n[5/7] Extracting meeting information...")

    action_items = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)

    questions = extract_questions(transcript)

    # --------------------------------
    # STEP 6: Build RAG
    # --------------------------------

    print("\n[6/7] Building RAG system...")

    rag_chain = build_rag_chain(transcript)

    print("\n[7/7] Pipeline completed successfully!")

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    # --------------------------------
    # Get user input
    # --------------------------------

    source = input(
        "Enter YouTube URL or local file path: "
    ).strip()

    language = input(
        "Language (english/hinglish): "
    ).strip() or "english"

    # --------------------------------
    # Run complete pipeline
    # --------------------------------

    result = run_pipeline(source, language)

    # --------------------------------
    # Display Results
    # --------------------------------

    print("\n" + "=" * 60)
    print(f"📌 Title:\n{result['title']}")

    print(f"\n📋 Summary:\n{result['summary']}")

    print(f"\n✅ Action Items:\n{result['action_items']}")

    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")

    print(f"\n❓ Open Questions:\n{result['open_questions']}")

    print("=" * 60)

    # --------------------------------
    # Chat with VideoMind
    # --------------------------------

    print("\n💬 Chat with your video")
    print("Type 'exit' to quit.\n")

    rag_chain = result["rag_chain"]

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(
            rag_chain,
            question
        )

        print(f"\n🤖 Assistant: {answer}\n")