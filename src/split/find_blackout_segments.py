"""
detect_blackouts.py
---

Detects blackout (dark) frame segments in videos and exports them to CSV.
"""

from dotenv import load_dotenv
from pathlib import Path
import os

import imageio
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

VIDEOS_DIR = Path(os.getenv("VIDEOS_DIR")).resolve()
BLACKOUTS_DIR = Path(os.getenv("BLACKOUTS_DIR")).resolve()

assert VIDEOS_DIR.is_dir(), "Could not find video source directory."
assert VIDEOS_DIR != BLACKOUTS_DIR, "Video directory and output directory for blackout segment CSV files must differ."

def merge_dark_segments(segments: list[dict], max_gap: int) -> list[dict]:
    """Merge dark segments if the gap between them is <= max_gap frames."""

    if not segments:
        return []

    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start_frame"] - prev["end_frame"] - 1

        if gap <= max_gap:
            prev["end_frame"] = seg["end_frame"]
            prev["end_time_ms"] = seg["end_time_ms"]
            prev["num_frames"] = prev["end_frame"] - prev["start_frame"] + 1
        else:
            merged.append(seg)

    return merged


def convert_ms_to_mmss(milliseconds: int) -> str:
    """Converts milliseconds to MM:SS format."""

    seconds, _ = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def detect_blackout_frames(
    video_path: Path,
    output_csv: Path,
    min_dark_frames: int = 5,
    merge_gap_frames: int = 5,
) -> None:
    """Detect blackout frame segments in a single video."""

    reader = imageio.get_reader(str(video_path), "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", None)
    print(f"Processing video: {video_path}")

    dark_events = []
    in_dark_segment = False
    segment_start = None
    frame_idx = 0

    for frame in reader:
        frame_idx += 1
        
        frame = np.asarray(frame, dtype=np.float32)

        # Crop borders (remove possible digitization borders)
        h, w = frame.shape[:2]
        frame = frame[h // 10 : 9 * h // 10, w // 10 : 9 * w // 10]

        # Luma calculation
        luma = (
            0.2126 * frame[..., 0]
            + 0.7152 * frame[..., 1]
            + 0.0722 * frame[..., 2]
        )
        luma = cv2.GaussianBlur(luma, (5, 5), 0)

        low = np.percentile(luma, 5)
        high = np.percentile(luma, 95)
        is_dark = (high - low) < 5.0

        if is_dark and not in_dark_segment:
            in_dark_segment = True
            segment_start = frame_idx

        elif not is_dark and in_dark_segment:
            segment_end = frame_idx - 1

            if segment_end - segment_start + 1 >= min_dark_frames:
                dark_events.append(
                    {
                        "start_frame": segment_start,
                        "end_frame": segment_end,
                        "start_time_ms": int(segment_start / fps * 1000) if fps else None,
                        "end_time_ms": int(segment_end / fps * 1000) if fps else None,
                        "num_frames": segment_end - segment_start + 1,
                    }
                )

            in_dark_segment = False
            segment_start = None

    # Handle video ending during a dark segment
    if in_dark_segment:
        segment_end = frame_idx
        dark_events.append(
            {
                "start_frame": segment_start,
                "end_frame": segment_end,
                "start_time_ms": int(segment_start / fps * 1000) if fps else None,
                "end_time_ms": int(segment_end / fps * 1000) if fps else None,
                "num_frames": segment_end - segment_start + 1,
            }
        )

    if not dark_events:
        print(f"{video_path.name}: no dark frames detected")
        return

    # Merge close segments
    dark_events = merge_dark_segments(dark_events, merge_gap_frames)

    for event in dark_events:
        event["start_time"] = convert_ms_to_mmss(event.pop("start_time_ms"))
        event["end_time"] = convert_ms_to_mmss(event.pop("end_time_ms"))

    df = pd.DataFrame(dark_events)
    df.to_csv(output_csv, index=False)
    print(f"{video_path.name}: detected {len(df)} dark segments")


def main():
    """Run blackout detection for exactly one video in VIDEOS_DIR."""
    
    BLACKOUTS_DIR.mkdir(parents=False, exist_ok=True)

    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    assert video_files, "No video files found."

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    print("SLURM_ARRAY_TASK_ID =", task_id)

    if task_id >= len(video_files):
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={task_id} but only "
            f"{len(video_files)} videos available."
        )

    video_path = video_files[task_id]
    output_csv = BLACKOUTS_DIR / f"{video_path.stem}_blackouts.csv"

    print(f"Processing video {task_id}/{len(video_files) - 1}: {video_path.name}")

    detect_blackout_frames(
        video_path=video_path,
        output_csv=output_csv,
    )

if __name__ == "__main__":
    main()
