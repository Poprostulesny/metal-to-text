# Data Labeler

Local Streamlit tool for cutting vocal-only song stems into short NeMo ASR
training segments.

## Requirements

The app expects system `ffmpeg` and `ffprobe` to be available on `PATH`.

## Install

From the repository root:

```bash
<ui-venv>/bin/python -m pip install -r data-labeler/requirements-labeler.txt
```

## Run

```bash
<ui-venv>/bin/streamlit run data-labeler/app.py
```

## Output

The app writes:

- WAV clips to `data-labeler/output/audio/`
- NeMo JSONL records to `data-labeler/output/segments.jsonl`
- auto-exported splits to `data/train_data_8.jsonl`, `data/test_data_1.jsonl`,
  and `data/valid_data_1.jsonl`

Each saved segment is a mono 16 kHz WAV and a manifest row with
`audio_filepath`, `duration`, `text`, `artist`, `source_audio_filepath`,
`source_start`, `source_end`, and `source_index`.

Reopening the app preserves previous work and appends new segments to the same
manifest. The undo button removes the latest segment for the current source song.
After every save or undo, the app rewrites the 80/10/10 train/test/valid split
in `data/`. The sidebar also has a manual export button.


## Smoke check

After saving at least one segment:

Use `ffprobe` on one of the generated clips:

```bash
ffprobe -v error -show_entries stream=sample_rate,channels -of default=nw=1 data-labeler/output/audio/<segment>.wav
```
