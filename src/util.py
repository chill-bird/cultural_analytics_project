"""
match_chapter_info.py
---

Merge information for chapters to one dataframe.
"""

from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import os

# Paths
load_dotenv()

VIDEO_DATA_TSV = Path(os.getenv("VIDEO_DATA_TSV")).resolve()
CHAPTERS_DATA_TSV = Path(os.getenv("CHAPTERS_DATA_TSV")).resolve()

assert VIDEO_DATA_TSV.is_file(), "Could not find TSV."
assert CHAPTERS_DATA_TSV.is_file(), "Could not find TSV."


def frames_to_timestamp(frame_no: int, fps: int | float) -> str:
    """Convert a frame number to nn:ss timestamp format in movie"""
    secs = round(frame_no / fps, 0)
    return seconds_to_mmss(secs)


def frames_to_millisecs(frame_no: int, fps: int | float) -> float:
    """Convert a frame number to nn:ss timestamp format in movie"""
    return round((frame_no / fps) * 1000, 0)


def mmss_to_ms(mmss: str | None) -> int | None:
    """
    Convert a 'mm:ss' time string to total milliseconds.

    Example:
        '02:30' -> 150
    """
    if mmss in [None, ""] or (not isinstance(mmss, str) and mmss.isna()):
        return None
    try:
        minutes, seconds = mmss.split(":")
        return int(minutes) * 60 * 1000 + int(seconds) * 1000
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid time format: {mmss!r}. Expected 'mm:ss'.")


def seconds_to_mmss(secs: int | float | None) -> str:
    """
    Convert total seconds to mm:ss format.

    Example:
        150 ->  '02:30'
    """
    if secs in [None, ""]:
        return ""
    try:
        total_seconds = int(secs)
        if total_seconds < 0:
            total_seconds = 0

        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid time format: {secs!r}. Expected 'mm:ss'.")


def get_chapter_mapping_df(filestem: str | None) -> pd.DataFrame | None:

    if not filestem:
        return None
    CHAPTERS_TIMESTAMPED_DIR = Path(os.getenv("CHAPTERS_TIMESTAMPED_DIR")).resolve()
    assert CHAPTERS_TIMESTAMPED_DIR.is_dir(), "Could not find directory for timestamped chapters."

    chapters_mapping_csv = Path(CHAPTERS_TIMESTAMPED_DIR / (filestem + ".csv")).resolve()
    if not chapters_mapping_csv.is_file():
        return None
    df_chapters = pd.read_csv(chapters_mapping_csv, sep=",")
    return df_chapters


def get_chapters_df(filestem: str | None) -> pd.DataFrame | None:

    if not filestem:
        return None
    CHAPTERS_DIR = Path(os.getenv("CHAPTERS_DIR")).resolve()
    assert CHAPTERS_DIR.is_dir(), "Could not find chapters directory."

    chapters_tsv = Path(CHAPTERS_DIR / (filestem + "_chapters.tsv")).resolve()
    if not chapters_tsv.is_file():
        return None
    df_chapters = pd.read_csv(chapters_tsv, sep="\t")
    return df_chapters


def get_transcription_df(filestem: str | None) -> pd.DataFrame | None:

    if not filestem:
        return None
    TRANSCRIPTIONS_DIR = Path(os.getenv("TRANSCRIPTIONS_DIR")).resolve()
    assert TRANSCRIPTIONS_DIR.is_dir(), "Could not find transcriptions directory."

    transcription_tsv = Path(TRANSCRIPTIONS_DIR / (filestem + ".tsv")).resolve()
    if not transcription_tsv.is_file():
        return None
    df_transcriptions = pd.read_csv(transcription_tsv, sep="\t")
    return df_transcriptions


def get_content_flags_df(filestem: str | None, chapter_name: str) -> pd.DataFrame | None:

    if not filestem or not chapter_name:
        return None
    CHAPTERS_FLAGGED_DIR = Path(os.getenv("CHAPTERS_FLAGGED_DIR")).resolve()
    assert CHAPTERS_FLAGGED_DIR.is_dir(), "Could not find directory for flagged content chapters."

    chapter_name = str(chapter_name)
    flags_csv = Path(CHAPTERS_FLAGGED_DIR / (filestem + "_" + chapter_name + ".csv")).resolve()
    if not flags_csv.is_file():
        return None
    df_flags = pd.read_csv(flags_csv, sep=",")
    return df_flags


def get_scenes_df(filestem: str | None) -> pd.DataFrame | None:

    if not filestem:
        return None
    SCENES_DIR = Path(os.getenv("SCENES_DIR")).resolve()
    assert SCENES_DIR.is_dir(), "Could not find directory for flagged content chapters."

    scenes_txt = Path(SCENES_DIR / (filestem + ".mp4.scenes.txt")).resolve()
    if not scenes_txt.is_file():
        return None

    df = pd.read_csv(scenes_txt, sep=" ", names=["start_frame", "end_frame"])
    df["start"] = df["start_frame"].apply(lambda x: frames_to_millisecs(x, 25))
    df["end"] = df["end_frame"].apply(lambda x: frames_to_millisecs(x, 25))
    # df["start_ts"] = df["start_frame"].apply(lambda x: frames_to_timestamp(x, 25))
    # df["end_ts"] = df["end_frame"].apply(lambda x: frames_to_timestamp(x, 25))
    return df
