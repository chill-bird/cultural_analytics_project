"""
convert_jsonl_to_csv.py
---

Reads all jsonl files within the NARRATIVES_DIR and creates one TSV file
"""
from dotenv import load_dotenv
import os
import json
from pathlib import Path
import csv

# Paths
load_dotenv()

INPUT_DIR = Path(os.getenv("NARRATIVES_DIR")).resolve()
assert INPUT_DIR.is_dir(), "Could not find directory for narratives detection results."
NARRATIVES_DATA_TSV = Path(os.getenv("NARRATIVES_DATA_TSV")).resolve()

rows = []

for file_path in INPUT_DIR.glob("*.jsonl"):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            analysis = data["analysis"]
            
            row = {
                "filestem": data["filestem"],
                "chapter": data["chapter"],
                "unit_of_analysis": analysis["unit_of_analysis"],
                "actor_configuration": ";".join(analysis["actor_configuration"]),
                "narrative_framing": ";".join(analysis["narrative_framing"]),
                "legitimation_strategy": ";".join(analysis["legitimation_strategy"]),
                "enemy_moral_status": ";".join(analysis["enemy_moral_status"]),
                "embodiment_mode": ";".join(analysis["embodiment_mode"]),
                "violence_visibility": ";".join(analysis["violence_visibility"]),
                "agency_level": analysis["agency_level"],
                "narrative_summary": analysis["narrative_summary"],
                "confidence_score": analysis["confidence_score"]
            }
            
            rows.append(row)

with open(NARRATIVES_DATA_TSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print("TSV file created:", NARRATIVES_DATA_TSV)