"""
merge_keyframe_contents.py
---

Merge content scores for keyframes to one dataframe.
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

# Paths
load_dotenv()

CONTENT_CLASSIFICATIONS_DIR = Path(os.getenv("CONTENT_CLASSIFICATIONS_DIR")).resolve()

assert (
    CONTENT_CLASSIFICATIONS_DIR.is_dir()
), "Could not find keyframe content classification directory."

TARGET_FILE = Path(os.getenv("CONTENT_DATA_TSV")).resolve()

COLUMNS_TO_KEEP = [
    "filestem",
    "chapter",
    "frame",
    "prediction",
    "score",
]


def main() -> None:

    dfs = []
    for p in CONTENT_CLASSIFICATIONS_DIR.glob("*.csv"):
        df = pd.read_csv(p)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    result = result[COLUMNS_TO_KEEP]

    result = result.drop_duplicates()
    result = result.sort_values(["filestem", "frame"])

    result.to_csv(TARGET_FILE, index=False, sep="\t")


if __name__ == "__main__":
    main()
