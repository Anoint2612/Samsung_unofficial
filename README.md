# Audio File Conversion and Human Voice Detection

This project provides a Python solution for:

1. **Converting audio files** from formats like `.mp3` and `.mp4` to `.wav`.
2. **Filtering audio files** based on whether they contain human voices (excluding non-human sounds like animals or musical instruments).

## Features

- Convert audio files (MP3, MP4, etc.) to WAV format using the `pydub` library.
- Analyze and classify the content of the audio using a pretrained deep learning model to determine if it contains human speech.
- The classification is done using the `speechbrain` library with a pretrained model for speaker recognition.

## Prerequisites

Make sure you have the following installed:

- Python 3.7 or later
- `ffmpeg` (required for audio format conversions)

### Install `ffmpeg`

- On Linux:
  ```bash
  sudo apt-get install ffmpeg
