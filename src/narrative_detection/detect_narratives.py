"""
detect_narratives.py
---

For each chapter in each video, detect narratives with LLM
"""

import os
import json
import time
import base64
import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Paths
load_dotenv()

# -----------------------
# Config
# -----------------------
MODEL = "gpt-4.1"
TEMPERATURE = 0.1
MAX_FRAMES_PER_CLIP = 8
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # exponential factor

CHAPTERS_TSV = Path(os.getenv("CHAPTERS_DATA_TSV")).resolve()
assert CHAPTERS_TSV.is_file(), "Could not find TSV."
# PROMPT_FILE = Path(Path(__file__).parent.resolve() / "prompt_detect_narratives.txt").resolve()
# assert PROMPT_FILE.is_file(), "Could not find prompt file."
SCHEMA_FILE = Path(os.getenv("SCHEMA_JSON")).resolve()
assert SCHEMA_FILE.is_file(), "Could not find schema file."
OUTPUT_DIR = Path(os.getenv("NARRATIVES_DIR")).resolve() 
KEYFRAMES_DIR = Path(os.getenv("KEYFRAMES_DIR")).resolve()
assert KEYFRAMES_DIR.is_dir(), "Could not find keyframes directory."

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing.")

# -----------------------
# Utilities
# -----------------------
def setup_logging(episode_id: str):
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/episode_{episode_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_schema():
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def select_keyframes(filestem: str, start_frame: int, end_frame: int) -> List[Path]:
    folder = Path(KEYFRAMES_DIR) / filestem

    if not folder.exists():
        return []

    all_frames = []

    for file in folder.glob("*.jpg"):
        try:
            frame_number = int(file.stem)
            if start_frame <= frame_number <= end_frame:
                all_frames.append((frame_number, file))
        except ValueError:
            continue

    # Sort by frame number
    all_frames.sort(key=lambda x: x[0])

    total_frames = len(all_frames)

    # Sample evenly if too many
    if len(all_frames) > MAX_FRAMES_PER_CLIP:
        step = len(all_frames) / MAX_FRAMES_PER_CLIP
        sampled = [
            all_frames[int(i * step)][1]
            for i in range(MAX_FRAMES_PER_CLIP)
        ]
    else:
        sampled = [f[1] for f in all_frames]

    logging.info(f"Found {total_frames} keyframes for {filestem}, selected {len(sampled)}: {[p.name for p in sampled]}")

    return sampled

def build_messages(row, schema, image_paths):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are a precise multimodal historical discourse analysis model."
                        "You specialize in narrative and visual analysis of propaganda media. "
                        "You must follow instructions exactly and return strictly valid JSON."
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"""
                    You are a historian analyzing Nazi propaganda newsreels (1940–1945).

                    Your task is to classify the narrative framing of a single video chapter.

                    The analysis MUST follow this internal procedure:

                    STEP 1 – VISUAL ANALYSIS (images only):
                    Analyze only the provided keyframes.
                    Identify:
                    - body representation (wounded, disciplined, invulnerable, etc.)
                    - presence or absence of violence
                    - depiction of enemy (if visible)
                    - age representation (youthful soldiers, adult soldiers, etc.)

                    STEP 2 – TEXTUAL ANALYSIS (audio transcription only):
                    Analyze only the audio transcription.
                    Identify:
                    - narrative framing (heroic combat, defensive war, etc.)
                    - legitimation logic (defensive, preventive, civilizing, etc.)
                    - enemy construction (criminalized, dehumanized, weaker enemy, etc.)
                    - positioning of soldiers (hero, victim, perpetrator, benefactor)

                    STEP 3 – MULTIMODAL INTEGRATION:
                    Compare visual and textual findings.
                    - Treat visual and textual material as analytically equal.
                    - If they reinforce each other, integrate them.
                    - If they contradict each other, note the contradiction internally.
                    - Do NOT privilege narration automatically over visual evidence.
                    - Consider meaningful absences (e.g., violence not shown).

                    FINAL STEP – CLASSIFICATION:
                    Classify strictly according to the provided schema.
                    Only use categories defined in the schema.
                    If evidence for a field is insufficient, use "none". Only assign a legitimation_strategy if the narration articulates justification. If no explicit justification is present, return “none”.
                    Allow multiple labels where logically appropriate.
                    Omit fields not present in the schema.
                    Return strictly valid JSON.
                    Do NOT explain your reasoning.
                    Do NOT include text outside the JSON.

                    SCHEMA:
                    {json.dumps(schema, indent=2)}

                    AUDIO TRANSCRIPTION (German, analyze in original language):
                    {row['audio_transcription']}
                    """
                }
            ] + [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_image(p)}"
                }
                for p in image_paths
            ]
        }
    ]

    return messages


def call_model(client, messages):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=MODEL,
                input=messages
            )

            # Extract text from multimodal response
            text_output = ""
            for message in response.output:
                # message.type is an attribute
                if message.type == "message":
                    for content in message.content:
                        if content.type == "output_text":
                            text_output += content.text

            return text_output

        except Exception as e:
            wait_time = RETRY_BACKOFF ** attempt
            logging.warning(f"API error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError("Max retries exceeded.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True, help="Episode ID")
    args = parser.parse_args()

    episode_id = args.episode

    client = OpenAI(api_key=OPENAI_API_KEY)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = Path(OUTPUT_DIR) / f"episode_{episode_id}.jsonl"

    schema = load_schema()

    df = pd.read_csv(CHAPTERS_TSV, sep="\t")

    # Get sorted list of unique episodes
    episodes = sorted(df["episode"].unique())

    # Map Slurm array task ID to actual episode
    slurm_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if slurm_id >= len(episodes):
        raise RuntimeError(f"SLURM_ARRAY_TASK_ID {slurm_id} out of range.")

    episode_id = episodes[slurm_id]
    
    setup_logging(episode_id)
    logging.info(f"Starting episode {episode_id}")

    df_episode = df[df["episode"] == episode_id]
    if df_episode.empty:
        raise RuntimeError(f"No rows found for episode {episode_id}.")
    
    # only keep chapters where german_soldiers_depicted is True
    df_episode = df_episode[df_episode["german_soldiers_depicted"] == True]

    if df_episode.empty:
        logging.info(f"No chapters with German soldiers in episode {episode_id}.")
        return

    # resume support
    processed = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                processed.add(entry["filestem"])

    logging.info(f"Already processed {len(processed)} clips.")

    with open(output_file, "a", encoding="utf-8") as outfile:

        for _, row in df_episode.iterrows():

            filestem = row["filestem"]
            chapter = row["chapter"] 

            if filestem in processed:
                continue

            logging.info(f"Processing filestem: {filestem}, chapter: {chapter}")

            image_paths = select_keyframes(
                filestem,
                int(row["start_frame"]),
                int(row["end_frame"])
            )

            messages = build_messages(row, schema, image_paths)

            try:
                result_text = call_model(client, messages)

                if not result_text.strip():
                    raise ValueError(f"Empty response from model for filestem {filestem}, chapter {chapter}")

                try:
                    result_json = json.loads(result_text)
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON for filestem {filestem}, chapter {chapter}: {result_text!r}")
                    failed_file = Path(OUTPUT_DIR) / "failed_responses.jsonl"
                    
                    with open(failed_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "filestem": filestem,
                            "chapter": chapter,
                            "raw_response": result_text
                        }, ensure_ascii=False) + "\n")

                    continue  
                
                output_entry = {
                    "filestem": filestem,
                    "chapter": chapter,
                    "analysis": result_json
                }

                outfile.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                outfile.flush()

                logging.info(f"Finished {filestem}")

            except Exception as e:
                logging.error(f"Failed {filestem}: {e}")

    logging.info("Episode complete.")


if __name__ == "__main__":
    main()
    