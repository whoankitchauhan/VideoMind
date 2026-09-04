import os
import subprocess
import yt_dlp

# ------------------------------------------------------------
# Download folder
# ------------------------------------------------------------

DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. Download audio from YouTube
# ------------------------------------------------------------

def download_youtube_audio(url: str) -> str:
    print("\n[1/3] Downloading audio from YouTube...")

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,

        # Convert downloaded audio to MP3 using FFmpeg
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],

        # Reduce unnecessary yt-dlp output
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        filename = ydl.prepare_filename(info)

        # yt-dlp may initially use .webm or .m4a
        # FFmpeg converts it to .mp3
        filename = (
            filename
            .replace(".webm", ".mp3")
            .replace(".m4a", ".mp3")
        )

    print(f"✓ Audio downloaded: {os.path.basename(filename)}")

    return filename


# ------------------------------------------------------------
# 2. Convert audio to WAV
# ------------------------------------------------------------

def convert_to_wav(input_file: str) -> str:
    print("[2/3] Converting audio to WAV format...")

    output_file = (
        os.path.splitext(input_file)[0]
        + "_converted.wav"
    )

    command = [
        "ffmpeg",

        # Input audio
        "-i", input_file,

        # Convert stereo → mono
        "-ac", "1",

        # Set sample rate to 16 kHz
        "-ar", "16000",

        # Overwrite existing file
        "-y",

        # Output file
        output_file
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(f"✓ WAV file created: {os.path.basename(output_file)}")

    return output_file


# ------------------------------------------------------------
# 3. Split WAV into smaller chunks
# ------------------------------------------------------------

def chunk_audio(
    wav_path: str,
    chunk_length_minutes: int = 10
) -> list:
    print(
        f"[3/3] Splitting audio into "
        f"{chunk_length_minutes}-minute chunks..."
    )

    output_pattern = (
        f"{os.path.splitext(wav_path)[0]}_chunk_%03d.wav"
    )

    # Convert minutes to seconds
    chunk_seconds = chunk_length_minutes * 60

    command = [
        "ffmpeg",
        "-i", wav_path,

        # Split the audio into segments
        "-f", "segment",

        # Length of each segment
        "-segment_time", str(chunk_seconds),

        # Copy audio without re-encoding
        "-c", "copy",

        # Start timestamps from zero for every chunk
        "-reset_timestamps", "1",

        # Overwrite existing chunks
        "-y",

        output_pattern
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    directory = os.path.dirname(wav_path)
    base_name = os.path.splitext(
        os.path.basename(wav_path)
    )[0]

    chunks = []

    for filename in sorted(os.listdir(directory)):
        if (
            filename.startswith(base_name + "_chunk_")
            and filename.endswith(".wav")
        ):
            chunks.append(
                os.path.join(directory, filename)
            )

    print(f"✓ {len(chunks)} audio chunks created.")

    return chunks


# ------------------------------------------------------------
# Main audio processing function
# ------------------------------------------------------------

def process_input(source: str) -> list:
    print("\n" + "=" * 50)
    print("        VideoMind Audio Processing")
    print("=" * 50)

    # Check whether the input is a URL or local file
    if source.startswith(("http://", "https://")):

        print("Source: YouTube URL")

        # YouTube URL → MP3
        audio_file = download_youtube_audio(source)

        # MP3 → WAV
        wav_file = convert_to_wav(audio_file)

    else:

        print("Source: Local audio file")

        # Local file → WAV
        wav_file = convert_to_wav(source)

    # WAV → smaller chunks
    chunks = chunk_audio(wav_file)

    print("\n" + "=" * 50)
    print("        Audio Processing Complete")
    print("=" * 50)
    print(f"Total chunks: {len(chunks)}")
    print(f"Output folder: {DOWNLOAD_DIR}")
    print("=" * 50 + "\n")

    return chunks
