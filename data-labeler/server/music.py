"""Reading song metadata from music/music.json.

A "song" here is one entry in music.json. Its raw ``path`` points at the
originally downloaded file; the vocal stem we actually label lives in
music_finder/final_music under the same filename.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or "segment"


@lru_cache(maxsize=1)
def load_songs() -> list[dict[str, Any]]:
    if not config.MUSIC_JSON.exists():
        raise FileNotFoundError(f"Missing metadata file: {config.MUSIC_JSON}")
    with config.MUSIC_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {config.MUSIC_JSON}")
    return data


def song_count() -> int:
    return len(load_songs())


def get_song(index: int) -> dict[str, Any]:
    songs = load_songs()
    if index < 0 or index >= len(songs):
        raise IndexError(f"Song index out of range: {index}")
    return songs[index]


def vocal_stem_path(song: dict[str, Any]) -> Path:
    raw_path = Path(str(song.get("path", "")))
    return config.VOCAL_DIR / raw_path.name


def song_id(song: dict[str, Any]) -> str:
    """Stable identifier used for filenames, VAD cache and UI state keys."""
    return _slug(Path(str(song.get("path", ""))).stem)


def song_title(song: dict[str, Any], index: int) -> str:
    stem = Path(str(song.get("path", ""))).stem
    return stem or f"Song {index + 1}"


def song_summary(index: int) -> dict[str, Any]:
    """Lightweight payload for the song list / current-song header."""
    song = get_song(index)
    stem = vocal_stem_path(song)
    return {
        "index": index,
        "id": song_id(song),
        "title": song_title(song, index),
        "artist": str(song.get("artist") or "Unknown artist"),
        "genre": song.get("genre") or [],
        "lyrics": str(song.get("lyrics") or ""),
        "vocal_path": str(stem),
        "has_vocal": stem.exists(),
    }
