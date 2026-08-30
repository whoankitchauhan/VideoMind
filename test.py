from utlis.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks


source = "https://youtu.be/Lg-meK5IU8Q?si=rbeFvGQAFxXdsKhM"


print("\n" + "=" * 50)
print("           VideoMind Test")
print("=" * 50)


# Step 1: Download, convert and chunk audio
chunks = process_input(source)


# Step 2: Transcribe all audio chunks using Whisper
transcription = transcribe_audio_chunks(
    chunks,
    translate=False
)


# Step 3: Display final transcription
print("\n" + "=" * 50)
print("        Final Transcription")
print("=" * 50)

print(transcription)

print("\n" + "=" * 50)
print("        Transcription Complete")
print("=" * 50)