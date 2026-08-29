import whisper 
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
_model = None

def load_whisper_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model
