# Data Labeler

Fast, local tool for cutting vocal-only song stems into short NeMo ASR training
segments. Waveform-first: Silero VAD proposes regions, you fine-tune them by
drag, and you grab the text straight from the lyrics by selecting it.

## Requirements

System `ffmpeg` and `ffprobe` on `PATH`. Everything else installs into an
isolated `data-labeler/.venv` on first launch (small: FastAPI + onnxruntime,
no torch). The Silero VAD ONNX model downloads automatically into `models/`.

## Run

Windows:

```bat
data-labeler\run.bat
```

Linux / macOS:

```bash
data-labeler/run.sh
```

First launch creates the venv, installs deps and opens
<http://127.0.0.1:8765>. Later launches start instantly. Override the port with
`LABELER_PORT` (default 8765 — the 8000 range is often Hyper-V-reserved on
Windows).

## Workflow

1. Pick a song (top bar). The waveform loads and Silero VAD regions appear; the
   next few songs are pre-computed in the background so switching is instant.
2. Click a VAD region (or drag your own) to set the **active** selection. Drag
   the handles to fine-tune.
3. Select the matching text in the **Lyrics** panel — it fills the segment text
   and is greyed out once saved, with a cursor marking where you left off.
4. **Save**. The clip and manifest row are written and the splits rebuilt.

Click a saved segment to edit its bounds/text, or delete it.

### Shortcuts

| Key | Action |
| --- | --- |
| `Space` | play active region |
| `L` | toggle loop |
| `Enter` / `Ctrl+Enter` | save segment |
| `Tab` / `Shift+Tab` | next / previous VAD region |
| `←` `→` | nudge region end (`Shift` = bigger step) |
| `Alt`+`←` `→` | nudge region start |
| `Del` / `Backspace` | delete the segment being edited |
| `Ctrl+Z` | undo last saved segment |

## Output

The manifest format is unchanged and stays 1:1 for NeMo:

- WAV clips → `data-labeler/output/audio/` (mono 16 kHz)
- master manifest → `data-labeler/output/segments.jsonl`
- rebuilt splits → `data/train_data_8.jsonl`, `data/test_data_1.jsonl`,
  `data/valid_data_1.jsonl` (seeded 80/10/10, rebuilt after every change)

Each manifest row: `audio_filepath`, `duration`, `text`, `artist`, `genre`,
`source_audio_filepath`, `source_start`, `source_end`, `source_index`.

UI-only state (which lyric range each segment used) lives separately in
`data-labeler/output/label_state.json`, so the manifest stays clean.

## Architecture

```
server/   FastAPI backend
  config.py    paths + constants
  music.py     read music.json, map vocal stems
  audio.py     ffmpeg/ffprobe cut + probe
  manifest.py  master segments.jsonl + derived splits (CRUD)
  state.py     label_state.json sidecar (lyric spans)
  vad.py       Silero VAD via onnxruntime, disk cache + prefetch
  main.py      HTTP routes
web/      wavesurfer.js frontend (no build step)
```
