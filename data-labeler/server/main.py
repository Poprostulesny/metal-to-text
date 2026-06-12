"""FastAPI app: thin HTTP layer over the domain modules.

Routes only validate input, call into music/audio/manifest/state/vad and shape
the JSON. All real logic lives in those modules.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from . import audio, config, lrc, manifest, music, state, vad
from .schemas import LrcOffset, RejectedKey, SegmentCreate, SegmentDelete, SegmentUpdate

app = FastAPI(title="Metal-to-text labeler")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((config.WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/config")
def get_config() -> dict:
    return {
        "max_segment_sec": config.DEFAULT_MAX_SEGMENT_SEC,
        "target_sample_rate": config.TARGET_SAMPLE_RATE,
    }


@app.get("/api/songs")
def list_songs() -> list[dict]:
    return [
        {k: music.song_summary(i)[k]
         for k in ("index", "id", "title", "artist", "has_vocal")}
        for i in range(music.song_count())
    ]


@app.get("/api/songs/{index}")
def get_song(index: int) -> dict:
    _guard_index(index)
    summary = music.song_summary(index)
    summary["duration"] = _safe_duration(summary["vocal_path"], summary["has_vocal"])
    summary["splits"] = _split_counts()
    summary["lrc_lines"] = lrc.lines_for_song(index, summary["duration"])
    summary["lrc_offset"] = lrc.get_offset(index)
    summary["rejected"] = state.rejected_for_song(summary["id"])
    return summary


@app.post("/api/songs/{index}/rejected")
def add_rejected(index: int, body: RejectedKey) -> dict:
    _guard_index(index)
    keys = state.add_rejected(music.song_id(music.get_song(index)), body.key)
    return {"rejected": keys}


@app.delete("/api/songs/{index}/rejected")
def clear_rejected(index: int) -> dict:
    _guard_index(index)
    state.clear_rejected(music.song_id(music.get_song(index)))
    return {"rejected": []}


@app.put("/api/songs/{index}/lrc_offset")
def set_lrc_offset(index: int, body: LrcOffset) -> dict:
    _guard_index(index)
    lrc.set_offset(index, body.offset)
    summary = music.song_summary(index)
    duration = _safe_duration(summary["vocal_path"], summary["has_vocal"])
    return {"offset": lrc.get_offset(index),
            "lrc_lines": lrc.lines_for_song(index, duration)}


@app.get("/api/songs/{index}/audio")
def get_audio(index: int) -> FileResponse:
    _guard_index(index)
    path = music.vocal_stem_path(music.get_song(index))
    if not path.exists():
        raise HTTPException(404, f"Vocal stem not found: {path}")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/songs/{index}/vad")
def get_vad(index: int) -> dict:
    _guard_index(index)
    regions = vad.regions_for_song(index)
    vad.prefetch(index)
    return {"regions": regions}


@app.get("/api/songs/{index}/segments")
def get_segments(index: int) -> list[dict]:
    _guard_index(index)
    spans = {s["audio_filepath"]: s for s in state.spans_for_song(index)}
    rows = []
    for row in manifest.rows_for_song(index):
        span = spans.get(row["audio_filepath"], {})
        rows.append({
            **row,
            "lyric_start": span.get("lyric_start", -1),
            "lyric_end": span.get("lyric_end", -1),
        })
    return rows


@app.get("/api/preview")
def preview(index: int = Query(...), start: float = Query(...), end: float = Query(...)) -> Response:
    _guard_index(index)
    path = music.vocal_stem_path(music.get_song(index))
    if not path.exists():
        raise HTTPException(404, "Vocal stem not found")
    return Response(audio.cut_to_bytes(path, start, end), media_type="audio/wav")


@app.post("/api/segments")
def create_segment(body: SegmentCreate) -> dict:
    _guard_index(body.song_index)
    row = manifest.add_segment(body.song_index, body.start, body.end, body.text)
    if body.lyric_start >= 0 and body.lyric_end > body.lyric_start:
        state.set_span(row["audio_filepath"], body.song_index, body.lyric_start, body.lyric_end)
    return row


@app.put("/api/segments")
def edit_segment(body: SegmentUpdate) -> dict:
    row = manifest.update_segment(body.audio_filepath, body.start, body.end, body.text)
    song_index = int(row.get("source_index", 0))
    if body.lyric_start >= 0 and body.lyric_end > body.lyric_start:
        state.set_span(body.audio_filepath, song_index, body.lyric_start, body.lyric_end)
    else:
        state.remove_span(body.audio_filepath)
    return row


@app.delete("/api/segments")
def remove_segment(body: SegmentDelete) -> dict:
    row = manifest.delete_segment(body.audio_filepath)
    state.remove_span(body.audio_filepath)
    return {"deleted": row["audio_filepath"]}


# --- helpers ---------------------------------------------------------------

def _guard_index(index: int) -> None:
    if index < 0 or index >= music.song_count():
        raise HTTPException(404, f"Song index out of range: {index}")


def _split_counts() -> dict[str, int]:
    train, test, valid = manifest.split_rows(manifest.read_rows())
    return {"train": len(train), "test": len(test), "valid": len(valid)}


def _safe_duration(vocal_path: str, has_vocal: bool) -> float:
    if not has_vocal:
        return 0.0
    try:
        return round(audio.duration_sec(vocal_path), 3)
    except Exception:
        return 0.0


# Static assets (js/css) served from /static.
app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")


def run() -> None:
    import os

    import uvicorn

    if not audio.ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe must be available on PATH.")
    # 8765 by default: on Windows the 8000 range is often reserved by Hyper-V
    # and bind fails with WinError 10013. Override with LABELER_PORT.
    port = int(os.environ.get("LABELER_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port)
