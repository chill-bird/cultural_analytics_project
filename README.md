# README

## Project structure

```sh
.
├── dat/  # Datasets directory
│   ├── blackout_data/  # Results from black out analysis
│   ├── chapter_data.tsv  # Chapter data
│   └── video_data.tsv  # Video data
├── doc/  # Documentation files
└── src/  # Source code
    ├── analysis/
    ├── chapters/  # Merge chapter information from several sources
    └── transcription/  # Transcribe video audio data
```

## Data

Due to repository size limits, raw data were deleted from repository and, instead, aggregated in the two following `tsv` files.

1. `video_data.tsv`

    - **episode**: Episode number
    - **year**: Year
    - **filestem**: Filestem as ID
    - **length**: Length in seconds
    - **fps**: Frames per Second
    - **frames**: Number of frames
    - **titel**: Title of the episode
    - **url_bundesarchiv**: URL to movie source
    - **keywords**: Content keywords from movie source
    - **has_transcription**: Has an audio transcription file
    - **has_chapters**: Has a chapters file

2. `chapter_data.tsv`
    - **titel**: Title of the episode
    - **chapter**: Chapter title
    - **start_mm:ss**: Start timestamp in mm:ss format
    - **end_mm:ss**: End timestamp in mm:ss format
    - **is_war_report**: Chapter is flagged as war report
    - **is_combat_scene**: Chapter is flagged as combat scene
    - **german_soldiers_depicted**: Chapter is flagged as depicting German Soldiers
    - **shot_count**: Number of shots in this chapter
    - **content**: Chapter content description
    - **audio_transcription**: Audio transcription
    - **filestem**: Filestem from Video ID
    - **episode**: Episode number
    - **year**: Year
    - **start**: Start timestamp in milliseconds
    - **end**: End timestamp in milliseconds

### Chapter flags

`is_war_report`: The scene primarily focuses on reporting on the ongoing war and German troops are involved.
`is_combat_scene`: The scene shows actual combat with German participation (staged or re-inacted does count into it). Scenes like this depict German weapons and/or German soldiers in action. Combat scenes without German involvement do NOT count.
`german_soldiers_depicted`: The scene shows German soldiers. This can be combat or interaction with civilians or marches or anything more.

## Requirements

- Python 3.11 to 3.13

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

### Classify chapters with OpenAI

Prompts GPT to align chapters with audio transcription elements. Before starting script, an API key at OpenAI is required: <https://platform.openai.com/api-keys> and to be placed inside `.env` file.

```sh
python -m src.chapters.flag_chapters
```

### Merge chapter info

Merges information from timestamps and chapter descriptions to one TSV file `CHAPTERS_DATA_TSV`.

```sh
python -m src.chapters.merge_chapter_info
```

## Acknowledgements

Computations for this work were done (in part) using resources of the Leipzig University Computing Center.
