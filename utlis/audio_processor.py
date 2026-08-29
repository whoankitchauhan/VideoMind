import os
import subprocess
import yt_dlp


DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        filename = (
            filename
            .replace(".webm", ".mp3")
            .replace(".m4a", ".mp3")
        )

    return filename


def convert_to_wav(input_file: str) -> str:
    output_file = os.path.splitext(input_file)[0] + "_converted.wav"

    command = [
        "ffmpeg",
        "-i", input_file,
        "-ac", "1",
        "-ar", "16000",
        "-y",
        output_file
    ]

    subprocess.run(command, check=True)

    return output_file


def chunk_audio(wav_path: str, chunk_length_minutes: int = 10) -> list:
    output_pattern = (
        f"{os.path.splitext(wav_path)[0]}_chunk_%03d.wav"
    )

    chunk_seconds = chunk_length_minutes * 60

    command = [
        "ffmpeg",
        "-i", wav_path,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        "-reset_timestamps", "1",
        "-y",
        output_pattern
    ]

    subprocess.run(command, check=True)

    directory = os.path.dirname(wav_path)
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    chunks = []

    for filename in sorted(os.listdir(directory)):
        if filename.startswith(base_name + "_chunk_") and filename.endswith(".wav"):
            chunks.append(os.path.join(directory, filename))

    return chunks


data = download_youtube_audio(
    "https://youtu.be/HdafI0t3sEY?si=F7dfccMfrMMClQF7"
)

data_final = convert_to_wav(data)

def process_input(source: str) -> list:
    if source.startswith("http") or source.startswith("https"):
        print("Downloading audio from YouTube...")
        audio_file = download_youtube_audio(source)
    else:
        print("Using local audio file...")
        audio_file = convert_to_wav(source)

    print("Converting audio to WAV format...")
    wav_file = convert_to_wav(audio_file)
    print("Chunking audio into smaller segments...")
    chunks = chunk_audio(wav_file)
    print(f"Audio processing complete. {len(chunks)} chunks created.")
    return chunks