# README

## Project structure

```sh
.
├── dat/  # Datasets directory
│   ├── chapters/  # Chapters from Bundesarchiv
│   ├── transcriptions/  # Audio transcriptions
│   └── video_data.tsv  # Data set description (tab-separated)
├── doc/  # Documentation files
└── src/  # Source code
    └── analysis/
```

## Requirements

- Python 3.12 or higher

## Getting Started

- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
- Install requirements `pip install -r requirements.txt`
- Copy `.env.example` and save as `.env`. Add environment values to environment variables.

## Data preprocessing

### Transcribe audio data via whisper

Transcribes .wav audio files in cluster job

```sh
python -m src.transcribe.transcribe
```

### Detect blackout frames in video data

Detects contiguous segments of dark (blackout) frames in video files and exports one CSV per video. CSV files are saved to `BLACKOUTS_DIR`.

```sh
python -m src.split.find_blackout_segments
```

### Get timestamps for chapters with OpenAI

Prompts GPT to align chapters with audio transcription elements. Before starting script, an API key at OpenAI is required: <https://platform.openai.com/api-keys> and to be placed inside `.env` file.

```sh
python -m src.chapters.match_chapters_gpt
```

### Merge chapter info

Merges information from timestamps and chapter descriptions to one TSV file `CHAPTERS_DATA_TSV`.

```sh
python -m src.chapters.merge_chapter_info
```

### Select kayframes with LMSKE
https://github.com/ttharden/Keyframe-Extraction-for-video-summarization

1. Run TransNetV2 https://github.com/soCzech/TransNetV2
- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
```sh
pip install tensorflow==2.5.0
pip install ffmpeg-python pillow
pip install torch torchvision torchaudio

module load FFmpeg
git clone https://github.com/soCzech/TransNetV2.git
```

## Acknowledgements

Computations for this work were done (in part) using resources of the Leipzig University Computing Center.
