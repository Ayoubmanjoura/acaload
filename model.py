# Acaload - YouTube Stem Extractor
# Copyright (C) 2025 Ayoub Manjoura
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
import torchaudio
import os
import tempfile
import shutil
import logging
import yt_dlp
import argparse
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade

# Use "soundfile" backend (required for WAV saving/loading)
torchaudio.set_audio_backend("soundfile")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and setup device
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
sample_rate = bundle.sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Using sample rate: {sample_rate}, device: {device}")


def download_youtube_audio(url, output_path="."):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            audio_path = os.path.splitext(filename)[0] + ".wav"
            logger.info(f"Downloaded and converted to WAV: {audio_path}")
            return audio_path
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None


def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
    device = mix.device if device is None else torch.device(device)
    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment)
    overlap_frames = int(sample_rate * overlap)
    step = chunk_len - overlap_frames

    fade_in = Fade(fade_in_len=overlap_frames, fade_out_len=0)
    fade_out = Fade(fade_in_len=0, fade_out_len=overlap_frames)

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    start = 0
    while start < length:
        end = min(start + chunk_len, length)
        chunk = mix[:, :, start:end]

        with torch.no_grad():
            out = model(chunk)

        # Smooth transitions
        if start == 0:
            out = fade_out(out)
        elif end == length:
            out = fade_in(out)
        else:
            out = fade_in(fade_out(out))

        final[:, :, :, start:end] += out
        start += step

    return final


def save_audio(tensor, sample_rate, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torchaudio.save(file_path, tensor.cpu(), sample_rate)
    logger.info(f"Saved: {file_path}")


def clean_temp_dir(temp_dir):
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


def create_stems_dir(base_dir="separated_sources"):
    i = 0
    while True:
        path = os.path.join(base_dir, "stems" if i == 0 else f"stems ({i})")
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        i += 1


def prompt_for_link():
    while True:
        url = input("Enter YouTube video URL: ").strip()
        if "youtube.com" in url or "youtu.be" in url:
            return url
        logger.warning("Invalid URL, try again.")


def main():
    parser = argparse.ArgumentParser(
        description="Acaload - Extract stems from YouTube audio."
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    args = parser.parse_args()

    video_url = args.url or prompt_for_link()
    temp_dir = tempfile.mkdtemp(prefix="yt_temp_")

    audio_path = download_youtube_audio(video_url, output_path=temp_dir)
    if not audio_path:
        logger.error("Download failed. Exiting.")
        clean_temp_dir(temp_dir)
        exit(1)

    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        logger.info(f"Resampling audio from {sr} to {sample_rate}")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    waveform = waveform.to(device)

    # Normalize
    mean = waveform.mean(dim=1, keepdim=True)
    std = waveform.std(dim=1, keepdim=True) + 1e-9
    waveform_norm = (waveform - mean) / std

    logger.info("Separating sources...")
    sources = separate_sources(model, waveform_norm[None], device=device)[0]

    # Denormalize
    sources = sources * std.unsqueeze(0) + mean.unsqueeze(0)
    audios = dict(zip(model.sources, list(sources)))

    out_dir = create_stems_dir()
    for name, audio in audios.items():
        save_audio(audio.squeeze(0), sample_rate, os.path.join(out_dir, f"{name}.wav"))

    clean_temp_dir(temp_dir)
    logger.info("✅ Done! Check your separated stems in the folder.")


if __name__ == "__main__":
    main()
