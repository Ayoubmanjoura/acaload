import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

import matplotlib.pyplot as plt
from pytube import YouTube
from moviepy.editor import VideoFileClip
from IPython.display import Audio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import os
import tempfile

bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

sample_rate = bundle.sample_rate
print(f"Sample rate: {sample_rate}")

from torchaudio.transforms import Fade

def download_youtube_video(url, output_path='.'):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=False).first()  # Download video
        file_path = stream.download(output_path=output_path)
        return file_path
    except Exception as e:
        print(f"An error occurred while downloading the video: {e}")
        return None

def extract_audio_from_video(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()  # Ensure you close the video clip to free up resources
        return audio_path  # Return the path to the extracted audio on success
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")
        return None

def prompt_for_youtube_link():
    while True:
        url = input("Please enter the YouTube video URL: ")
        if "youtube.com" in url or "youtu.be" in url:
            return url
        else:
            print("Invalid YouTube URL. Please enter a valid YouTube video URL.")

def separate_sources(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
):
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

def plot_spectrogram(stft, title="Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    _, axis = plt.subplots(1, 1)
    axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.tight_layout()

def save_audio(tensor, sample_rate, file_path):
    output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(file_path, tensor.cpu(), sample_rate)
    print(f"Saved {file_path}")

def clean_temporary_files(temp_dir):
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
        os.rmdir(temp_dir)
        print(f"Deleted temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Failed to delete temporary files: {e}")

def create_unique_stems_directory(base_dir):
    i = 0
    while True:
        stems_dir = os.path.join(base_dir, f"stems" if i == 0 else f"stems ({i})")
        if not os.path.exists(stems_dir):
            os.makedirs(stems_dir)
            return stems_dir
        i += 1

# Prompt user for YouTube video URL
video_url = prompt_for_youtube_link()

# Create a temporary directory to store downloaded files
temp_dir = tempfile.mkdtemp(prefix='Temp_youtube_downloads_')

# Download the video file from YouTube
SAMPLE_VIDEO_PATH = download_youtube_video(video_url, output_path=temp_dir)
AUDIO_FILE_PATH = os.path.join(temp_dir, "audio.wav")

if SAMPLE_VIDEO_PATH:
    # Extract audio from the downloaded video
    AUDIO_FILE_PATH = extract_audio_from_video(SAMPLE_VIDEO_PATH, AUDIO_FILE_PATH)
    if AUDIO_FILE_PATH:
        waveform, sample_rate = torchaudio.load(AUDIO_FILE_PATH)
        waveform = waveform.to(device)
        mixture = waveform

        segment = 10
        overlap = 0.1

        print("Separating track")

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()

        sources = separate_sources(
            model,
            waveform[None],
            device=device,
            segment=segment,
            overlap=overlap,
        )[0]
        sources = sources * ref.std() + ref.mean()

        sources_list = model.sources
        sources = list(sources)

        audios = dict(zip(sources_list, sources))

        N_FFT = 4096
        N_HOP = 4
        stft = torchaudio.transforms.Spectrogram(
            n_fft=N_FFT,
            hop_length=N_HOP,
            power=None,
        )

        def output_results(predicted_source: torch.Tensor, source: str):
            plot_spectrogram(stft(predicted_source)[0], f"Spectrogram - {source}")
            return Audio(predicted_source, rate=sample_rate)

        segment_start = 150
        segment_end = 155

        frame_start = segment_start * sample_rate
        frame_end = segment_end * sample_rate

        # Create a unique stems directory
        output_dir = "separated_sources"
        stems_dir = create_unique_stems_directory(output_dir)

        for source_name, source_audio in audios.items():
            save_path = os.path.join(stems_dir, f"{source_name}.wav")
            save_audio(source_audio.squeeze(0), sample_rate, save_path)

        # Clean up: delete the downloaded MP3 and MP4 files
        clean_temporary_files(temp_dir)

    else:
        print("Failed to extract audio from the video.")
else:
    print("Failed to download the video.")

