"""
match_chapter_info.py
---

Merge information for chapters to one dataframe.
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from src.util import mmss_to_ms, get_chapter_mapping_df, get_chapters_df, get_transcription_df

# Paths
load_dotenv()

TSV = Path(os.getenv("VIDEO_DATA_TSV")).resolve()
CHAPTERS_DIR = Path(os.getenv("CHAPTERS_DIR")).resolve()
TRANSCRIPTIONS_DIR = Path(os.getenv("TRANSCRIPTIONS_DIR")).resolve()
CHAPTERS_TIMESTAMPED_DIR = Path(os.getenv("CHAPTERS_TIMESTAMPED_DIR")).resolve()
assert TSV.is_file(), "Could not find TSV."
assert CHAPTERS_DIR.is_dir(), "Could not find chapters directory."
assert TRANSCRIPTIONS_DIR.is_dir(), "Could not find transcriptions directory."
CHAPTERS_TIMESTAMPED_DIR.mkdir(parents=False, exist_ok=True)
assert CHAPTERS_TIMESTAMPED_DIR.is_dir(), "Could not find directory for timestamped chapters."

TARGET_FILE = Path(os.getenv("CHAPTERS_DATA_TSV")).resolve()

COLUMNS_TO_KEEP = [
    "title",
    "chapter",
    "start_mm:ss",
    "end_mm:ss",
    "content",
    "audio_transcription",
    "filestem",
    "episode",
    "year",
    "end",
    "start",
]


def get_transcription_for_chapter(
    df: pd.DataFrame, start: float | int | None, end: float | int | None
) -> str:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return ""
    mask = (df["start"] < end) & (df["end"] > start)
    texts = (
        df.loc[mask, "text"]
        .dropna()  # removes NaN / pd.NA
        .astype(str)  # ensures strings
        .str.strip()  # removes ""
    )

    texts = texts[texts != ""]  # drop empty strings

    return " ".join(texts)


def main() -> None:
    movies_df = pd.read_csv(TSV, sep="\t")
    # Only use data where transcription and chapters exist
    movies_df = movies_df[(movies_df["has_transcription"]) & (movies_df["has_chapters"])]
    
    n = len(movies_df)
    chapters_dfs = []
    for i, row in movies_df.iterrows():

        print(f"[{i+1}/{n}]{row["title"]}\n")

        # Ground truth: Chapters
        # For each movie, search chapters
        chapters_df = get_chapters_df(row["filestem"])
        chapters_df["title"] = row["title"]
        chapters_df["filestem"] = row["filestem"]
        chapters_df["episode"] = row["episode"]
        chapters_df["year"] = row["year"]
        chapters_df["chapter"] = chapters_df["chapter"].astype(str)

        # Merge timestamp mapping to chapters
        timestamp_mapping_df = get_chapter_mapping_df(row["filestem"])
        if not timestamp_mapping_df.empty:
            timestamp_mapping_df["chapter"] = timestamp_mapping_df["chapter"].astype(str)
            chapters_df = chapters_df.merge(timestamp_mapping_df, on="chapter", how="left")
            chapters_df["start"] = chapters_df["start_mm:ss"].fillna("").apply(mmss_to_ms)
            chapters_df["end"] = chapters_df["end_mm:ss"].fillna("").apply(mmss_to_ms)

        else:
            print("No timestamp mapping found.")

        # Append transcriptions
        transcriptions_df = get_transcription_df(row["filestem"])
        chapters_df["audio_transcription"] = chapters_df.apply(lambda row: get_transcription_for_chapter(transcriptions_df, row["start"], row["end"]), axis=1)

        chapters_dfs.append(chapters_df)

    result = pd.concat(chapters_dfs)
    result = result[COLUMNS_TO_KEEP]
    result.to_csv(TARGET_FILE, index=False, sep="\t")


if __name__ == "__main__":
    main()
