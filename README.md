# Propagandistische Stilmittel in der Darstellung der Deutschen Wehrmacht in der Deutschen Wochenschau im Kriegsverlauf 1940-1945

**Q1**: Gibt es einen Zusammenhang zwischen den filmischen Gestaltungsmerkmalen der Deutschen Wochenschau und dem Kriegsgeschehen?

- Siegeswochenschauen: "langanhaltende Schwenks, die Vorliebe für Totalen und einen ruhiger Schnitt" (Bartels, 2010, S. 478), "kompakte Aufnahmen der Wehrmacht, welche Kraft und Stärke vermitteln sollten, um die Moral und Siegeszuversicht in den eigenen Reihen zu stärken" (Bartels, 2010, S. 248)
- ab 1941: "nun rasche Szenenwechsel, die stärkere Verwendung von Nahaufnahmen sowie ein Anwachsen des Kommentars, der die im Film gezeigte Situation näher erläutern musste" (Bartels, 2010, S. 478)

**Q2**: Wie wurden die Rollen der Wehrmacht in der Deutschen Wochenschau während des zweiten Weltkriegs dargestellt und propagandistischen Narrative wurden eingesetzt?

- "Das bestimmende Moment der Kriegsberichterstattung lag jetzt nicht mehr in der Betonung der scheinbar unaufhaltsamen Vormarsches de deutschen Wehrmacht, sondern in der Herausstellung soldatischen Heldentums" (Bartels, 2010, S. 489)

## Project structure

```sh
.
├── dat/
│   ├── chapter_data.tsv  # Aggregated chapter data
│   └── keyframe_content_data.tsv  # Aggregated keyframe classification data
│   └── video_data.tsv  # Aggregated video data
├── doc/  # Documentation files
└── src/  # Source code
    ├── analysis/  # Data analysis
    ├── chapters/  # Extract and merge chapter information from several sources
    ├── keyframes/  # Keyframe classification scripts
    ├── transcribe/  # Audio data transcription and clean-up scripts
    ├── split/  # Blackout cut detection scripts
    └── util.py  # Utility functions
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
    - **chapter**: Chapter number
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

3. `keyframe_content_data.tsv`
    - **filestem**: Filestem from Video ID
    - **chapter**: Chapter number
    - **frame**: Frame number
    - **prediction**: Predicted class
    - **score**: Similarity score

### Chapter flags

`is_war_report`: The scene primarily focuses on reporting on the ongoing war and German troops are involved.
`is_combat_scene`: The scene shows actual combat with German participation (staged or re-inacted does count into it). Scenes like this depict German weapons and/or German soldiers in action. Combat scenes without German involvement do NOT count.
`german_soldiers_depicted`: The scene shows German soldiers. This can be combat or interaction with civilians or marches or anything more.

## Requirements

- Python 3.11 to 3.13
- Git LFS

## Getting Started

- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
- Install requirements `pip install -r requirements.txt`
- Copy `.env.example` and save as `.env`. Add environment values to environment variables.

## Analysis

### Q1: Analyze formal features in combat scenes

```sh
python -m src.analysis.q1_formal_features
```

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

### Select kayframes with LMSKE

Extracts keyframe from video as described in <https://github.com/ttharden/Keyframe-Extraction-for-video-summarization> and saves them.

1. Run TransNetV2 <https://github.com/soCzech/TransNetV2>

- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
- Install requirements `pip install -r requirements_lmske.txt`

```sh
git clone https://github.com/soCzech/TransNetV2.git
```

Run `run_transnetv2.sh`

1. Run OpenCLIP
In the second step ChineseCLIP was swapped for on OpenCLIP model for better compatibility with german context.

```sh
git clone https://github.com/ttharden/Keyframe-Extraction-for-video-summarization.git
```

Run `run_clip.sh`

1. Extract Keyframes

- Reformat CLIP Feature Outputs with `reformat_features.py`
- Replace `init_center.py`in Keyframe-Extraction-for-video-summarization project with `/src/keyframe_selection_LMSKE/init_center.py` (contains bug fix)
- Replace `KMeans_improvment.py`in Keyframe-Extraction-for-video-summarization project with `/src/keyframe_selection_LMSKE/iKMeans_improvment.py` (contains bug fix)
- Run `run_keyframe_selection.sh`

### Classify shot scale

Classifies Shot scales in keyframes of chapters using CLIP.

```sh
python -m src.keyframes.classify_shot_scale
```

### Detect narratives

Detect narratives used in the videos and commentary using OpenAI GPT-4.1. Before starting script, an API key at OpenAI is required: <https://platform.openai.com/api-keys> and to be placed inside `.env` file.

```sh
python -m src.narrative_detetction.detect_narratives
```

## Acknowledgements

Computations for this work were done (in part) using resources of the Leipzig University Computing Center.
