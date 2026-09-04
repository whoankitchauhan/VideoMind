import os

import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
_model = None

def load_whisper_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = load_whisper_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task=task)
    print(f"Transcribing chunk: {chunk_path} (Task: {task})")
    return result["text"]


def transcribe_audio_chunks(chunks: list, translate: bool = False) -> str:
    full_transcription = ""
    for i, chunk in enumerate(chunks):
        print(f"[Transcribing chunk {i+1}/{len(chunks)}] {chunk}")
        transcription = transcribe_chunk(chunk, translate)
        full_transcription += transcription + " "
        print(f"✓ Chunk {i+1} transcribed.")
    return full_transcription.strip()
