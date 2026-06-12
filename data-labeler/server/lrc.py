"""Synced (LRC) lyrics parsed into timed line proposals for the labeler.

LRC text is looked up per song, in priority order:
1. a "synced_lyrics" field on the music.json entry (raw LRC text, the format
   spotdl's "synced" lyrics provider stores there)
2. music/lrc/<track>.lrc as saved by music_finder/check_synced_lyrics.py

Line text is normalized to the manifest convention (lowercase, apostrophes
kept, other punctuation stripped) so saved segments match the cleaned lyrics.

A per-song global offset (seconds) compensates for LRC synced against a
different edition of the track (longer intro etc.). Offsets live in their own
sidecar file, keyed by song id, so music.json and the manifest stay clean.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import config, music

TIMESTAMP_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")

# LRC stores only line starts; cap line length so a line right before an
# instrumental break does not swallow the whole break.
MAX_LINE_SEC = 12.0
MIN_LINE_SEC = 0.2


def _normalize(text: str) -> str:
    text = text.replace("’", "'").lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse(lrc_text: str) -> list[dict[str, Any]]:
    entries = []
    for raw_line in lrc_text.splitlines():
        stamps = list(TIMESTAMP_RE.finditer(raw_line))
        if not stamps:
            continue
        text = _normalize(raw_line[stamps[-1].end():])
        if not text:
            continue
        # A line may carry several timestamps (repeated chorus): one entry each.
        for stamp in stamps:
            frac = (stamp.group(3) or "0").ljust(3, "0")
            time = int(stamp.group(1)) * 60 + int(stamp.group(2)) + int(frac) / 1000
            entries.append({"time": time, "text": text})
    entries.sort(key=lambda e: e["time"])
    return entries


def _lrc_text(song: dict[str, Any]) -> str:
    text = str(song.get("synced_lyrics") or "")
    if text:
        return text
    path = config.LRC_DIR / f"{Path(str(song.get('path', ''))).stem}.lrc"
    return path.read_text(encoding="utf-8") if path.exists() else ""


# --- per-song global offset --------------------------------------------------

def _load_offsets() -> dict[str, float]:
    if not config.LRC_OFFSETS_PATH.exists():
        return {}
    try:
        with config.LRC_OFFSETS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def get_offset(index: int) -> float:
    return float(_load_offsets().get(music.song_id(music.get_song(index)), 0.0))


def set_offset(index: int, offset: float) -> None:
    offsets = _load_offsets()
    song_id = music.song_id(music.get_song(index))
    if offset:
        offsets[song_id] = round(float(offset), 3)
    else:
        offsets.pop(song_id, None)
    config.ensure_output_dirs()
    with config.LRC_OFFSETS_PATH.open("w", encoding="utf-8") as file:
        json.dump(offsets, file, ensure_ascii=False, indent=2)


def lines_for_song(index: int, duration: float) -> list[dict[str, Any]]:
    """Timed lyric lines as {start, end, text} region proposals."""
    entries = _parse(_lrc_text(music.get_song(index)))
    offset = get_offset(index)
    lines = []
    for i, entry in enumerate(entries):
        start = entry["time"] + offset
        next_start = (entries[i + 1]["time"] + offset if i + 1 < len(entries)
                      else start + MAX_LINE_SEC)
        end = min(next_start, start + MAX_LINE_SEC)
        start = max(0.0, start)
        if duration > 0:
            end = min(end, duration)
        if end - start >= MIN_LINE_SEC:
            lines.append({"start": round(start, 3), "end": round(end, 3), "text": entry["text"]})
    return lines
