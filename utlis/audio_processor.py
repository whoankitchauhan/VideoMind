import os
import yt_dlp

DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

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


data = download_youtube_audio("https://www.youtube.com/watch?v=7HSSR1n8dgc")


def convert_to_wav(input_file: str) -> str:
    output_file = os.path.splitext(input_file)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_file, format="wav")
    return output_file


print(convert_to_wav(data))

def chunk_audio(wav_path: str, chunk_length_ms: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_length_ms * 60 * 1000  # Convert minutes to milliseconds
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        chunk_filename = f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)
    return chunks