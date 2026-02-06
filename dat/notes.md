# Data

- 673 (1943) is only music
- trotz gleicher Quelle -> Videos entstammen verschiedenen Digitalisierungen -> Unterschiede wie:
  - Logo des Bundesarchivs im Bild zu sehen (siehe 511)
  - schwarzer Rand + Rauschen am Bildrand (von Tonspur?) (siehe 555)
  - manche Episoden mit Namen der Kameraleute am Anfang (siehe 555)

## Bundesarchiv Beschreibungen

- 44 Videos ohne Beschreibung des Bundesarchivs
- 30 mit Vorfilm, 148 ohne

## Blackouts

- einige Episoden ohne Blackouts

## Workflow

1. Extract all videos from list of URLs
2. Extract keywords and chapters from URLs
3. Extract chapters from column "content" to `CHAPTERS_DIR`
4. Create timestamped mapping to `CHAPTERS_TIMESTAMPED_DIR`
