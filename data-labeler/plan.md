# Data Labeler implementation plan

## Goal

Build a local Streamlit tool for manually creating short NeMo ASR training
segments from full songs. The tool reads song metadata from `music/music.json`,
uses vocal-only stems from `music_finder/final_music` by default, lets the user
preview and cut many segments from each source file, and writes both WAV clips
and NeMo JSONL manifest entries. Audio probing and cutting is handled by
system `ffprobe`/`ffmpeg` so the UI venv stays small.

## Inputs

- Metadata: `/home/mateusz/PycharmProjects/metal-to-text/music/music.json`
- Default source audio: `/home/mateusz/PycharmProjects/metal-to-text/music_finder/final_music/<raw filename>`
- Metadata fields expected per song:
  - `path`: original downloaded song path
  - `lyrics`: full sanitized lyrics text
  - `artist`: artist name
  - `genre`: optional genre list

The labeler maps a raw path like:

```text
music/Silent Planet - Offworlder.wav
```

to:

```text
music_finder/final_music/Silent Planet - Offworlder.wav
```

## Outputs

- Segment WAV files: `data-labeler/output/audio/`
- Manifest file: `data-labeler/output/segments.jsonl`
- Auto-exported training manifests:
  - `data/train_data_8.jsonl`
  - `data/test_data_1.jsonl`
  - `data/valid_data_1.jsonl`

Each segment is written as mono 16 kHz WAV using `ffmpeg`. Each manifest row is
a NeMo-compatible JSON object with extra source metadata:

```json
{
  "audio_filepath": "/abs/path/to/segment.wav",
  "duration": 23.42,
  "text": "segment text",
  "artist": "Artist",
  "source_audio_filepath": "/abs/path/to/full/vocal.wav",
  "source_start": 12.34,
  "source_end": 35.76,
  "source_index": 7
}
```

## UI workflow

1. Load `music/music.json`.
2. Show song selector in the sidebar.
3. Show a `Next song` button that advances to the next metadata entry.
4. For the active song:
   - show artist, source path, duration, and full lyrics;
   - show full audio player if the vocal stem exists;
   - show controls for `start_sec`, `end_sec`, and segment `text`;
   - allow previewing the selected range;
   - save the selected range as a new segment.
5. Show all already saved segments for the current source audio.
6. Allow undoing the latest saved segment for the current source audio.
7. Auto-export an 80/10/10 train/test/valid split after every save and undo.

## Validation

`Save segment` must reject:

- missing source audio;
- empty text;
- `start_sec < 0`;
- `end_sec <= start_sec`;
- ranges beyond the source duration.

If a target filename already exists, append/increment a counter so no segment is
overwritten.

Opening the app must never clear previous work. New segments are appended to the
existing manifest. Undo is the only operation that rewrites the manifest, and it
removes just the latest row for the current source audio plus its generated WAV
when the WAV is inside `data-labeler/output/audio/`.

The `data/` manifests are generated from `data-labeler/output/segments.jsonl`.
They are overwritten on each export so they always reflect the current labeler
dataset. Split shuffling uses a fixed seed for repeatability.

## Test commands

Install UI dependency in a separate UI venv:

```bash
<ui-venv>/bin/python -m pip install -r data-labeler/requirements-labeler.txt
```

Run app:

```bash
<ui-venv>/bin/streamlit run data-labeler/app.py
```

Smoke-check latest output:

```bash
ffprobe -v error -show_entries stream=sample_rate,channels -of default=nw=1 data-labeler/output/audio/<segment>.wav
```
