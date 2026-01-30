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

- Python 3.13 or higher

## Getting Started

- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
- Install requirements `pip install -r requirements.txt`
- Copy `.env.example` and save as `.env`. Add environment values to environment variables.

## Run tasks

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

## Acknowledgements

Computations for this work were done (in part) using resources of the Leipzig University Computing Center.
