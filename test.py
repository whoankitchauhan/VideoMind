from utlis.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks

source = "https://youtu.be/Lg-meK5IU8Q?si=rbeFvGQAFxXdsKhM"

chunks = process_input(source)
transcription = transcribe_audio_chunks(chunks, translate=False)
print("\n" + "=" * 50)
print("        Final Transcription")
print("=" * 50)
print(transcription)
