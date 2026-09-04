from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks
from core.analyzer import analyze_transcript


from core.rag_Engine import build_rag_chain, ask_question


# Load API keys and environment variables from .env
load_dotenv()


def should_translate(language: str) -> bool:
    normalized_language = language.strip().lower()
    if normalized_language in ("english", "en"):
        return False
    if normalized_language in ("hinglish", "hindi", "hi"):
        return True

    raise ValueError(
        "Unsupported language. Please choose 'english' or 'hinglish'."
    )


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

    transcript = transcribe_audio_chunks(
        chunks,
        translate=should_translate(language)
    )

    print(
        f"\nRaw transcription "
        f"(first 300 characters):\n{transcript[:300]}"
    )

    # --------------------------------
    # STEP 3: Analyze Transcript
    # --------------------------------

    print("\n[3/5] Analyzing transcript...")

    analysis = analyze_transcript(transcript)

    # --------------------------------
    # STEP 4: Build RAG
    # --------------------------------

    print("\n[4/5] Building RAG system...")

    rag_chain = build_rag_chain(transcript)

    print("\n[5/5] Pipeline completed successfully!")

    return {
        "title": analysis["title"],
        "transcript": transcript,
        "summary": analysis["summary"],
        "action_items": analysis["action_items"],
        "key_decisions": analysis["key_decisions"],
        "open_questions": analysis["open_questions"],
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
