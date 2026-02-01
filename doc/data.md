# Data

## Data Retrieval

- Wochenschau-Videos 1940-1945 von <https://digitaler-lesesaal.bundesarchiv.de/>
  - Videos enthalten Schlagworte und Inhaltsbeschreibungen ("Kapitel")
- Audio-Transkription (inkl. Timestamps) mit [Whisper](https://openai.com/index/whisper/)
- Zuordnung der Kapitel zu Start- und End-Timestamps über [GPT5-mini](<https://platform.openai.com/docs/models/gpt-5-mini>)

Ergebnis: 2108 Wochenschau-Kapitel mit Timestamps

## Data Filtering (geplant)

- Kapitel klassifizieren auf `military scenes` and `soldier scenes` via textueller Inhaltsbeschreibungen
