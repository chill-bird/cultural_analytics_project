# README

## Project structure

```sh
.
├── dat/  # Datasets directory
├── doc/  # Documentation files
└── src/  # Source code
    └── analysis/
```

## Requirements

- Python 13.13 or higher

## Getting Started

- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
- Install requirements `pip install -r requirements.txt`
- Copy `.env.example` and save as `.env`. Add environment values to environment variables.

## Run tasks

### Transcribe audio data via whisper

Unzip raw files and split data into train, test, val set. Split files and classname-index associations are saved to `DAT_DIR`.

```sh
python -m src.transcribe.transcribe
```

## Acknowledgements

Computations for this work were done (in part) using resources of the Leipzig University Computing Center.
