from dotenv import load_dotenv

load_dotenv()

from utlis.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)

source = "https://youtu.be/Lg-meK5IU8Q?si=rbeFvGQAFxXdsKhM"

print("\n" + "=" * 60)
print("                  VideoMind Test")
print("=" * 60)

# ------------------------------------------------------------
# Step 1: Download, convert and chunk audio
# ------------------------------------------------------------

chunks = process_input(source)

# ------------------------------------------------------------
# Step 2: Transcribe audio using Whisper
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              Starting Transcription")
print("=" * 60)

transcription = transcribe_audio_chunks(
    chunks,
    translate=False,
)

print("\n" + "=" * 60)
print("              Final Transcription")
print("=" * 60)

print(transcription)

# ------------------------------------------------------------
# Step 3: Extract Action Items
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              ACTION ITEMS")
print("=" * 60)

action_items = extract_action_items(transcription)

print(action_items)

# ------------------------------------------------------------
# Step 4: Extract Key Decisions
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              KEY DECISIONS")
print("=" * 60)

key_decisions = extract_key_decisions(transcription)

print(key_decisions)

# ------------------------------------------------------------
# Step 5: Extract Open Questions
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              OPEN QUESTIONS")
print("=" * 60)

questions = extract_questions(transcription)

print(questions)

# ------------------------------------------------------------
# Test Complete
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("             VideoMind Test Complete")
print("=" * 60)
print("✓ Audio processing")
print("✓ Whisper transcription")
print("✓ Action item extraction")
print("✓ Decision extraction")
print("✓ Question extraction")
print("=" * 60)
