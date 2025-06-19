# Acaload ReadME

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&labelColor=gray)](https://www.python.org/downloads/release/python-3130/)
[![yt_dlp](https://img.shields.io/badge/yt_dlp-2025.06.09-FF0000?style=flat&labelColor=gray)](https://github.com/yt-dlp/yt-dlp)
[![Torch](https://img.shields.io/badge/PyTorch-2.7.0-EF4B2F?style=flat&labelColor=gray&logo=pytorch&logoColor=white)](https://pytorch.org/)
![Python CI](https://github.com/Ayoubmanjoura/acaload/actions/workflows/python-ci.yml/badge.svg)

Acaload is an open-source project designed to locally extract the stems of any song directly from YouTube using AI — no uploads, no shady servers.

Most online stem extractors force you to first download the song (MP3, M4A, WAV, etc.) and then upload it to their site. Plus, many of these sites lock you behind a paywall or limit usage. Acaload cuts the middleman: it downloads and processes everything locally, so you keep control and don’t pay a dime.

### Features:
Download audio directly from YouTube
Seperate Stems (Drums, Bass, Vocals, Other)
Save stems as WAV files
Simple CLI interface
Cross-platform (Linux, Windows, Mac OS)

### Dependecies:
Python 3.13+
ffmpeg (Make sure it is accessible via PATH)
torch==2.7.0
torchaudio==2.7.0
yt-dlp>=2024.1.1

### Instalation
```
Bash
pip install .
```

### Usage

```
Bash
acaload #song_url
```
or
```
Bash
acaload
Enter YouTube video URL: https://...
```
### License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html) - see the [LICENSE](LICENSE) file for details.