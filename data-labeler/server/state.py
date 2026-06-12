"""UI sidecar state: which lyric character range each segment consumed.

Kept separate from segments.jsonl on purpose, so the NeMo manifest stays 1:1.
Keyed by ``audio_filepath`` (the same stable id manifest uses), so a span is
naturally tied to its segment through create / edit / delete.
"""
from __future__ import annotations

import json
from typing import Any

from . import config


def _load() -> dict[str, dict[str, Any]]:
    if not config.STATE_PATH.exists():
        return {}
    try:
        with config.STATE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("spans", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(spans: dict[str, dict[str, Any]]) -> None:
    config.ensure_output_dirs()
    with config.STATE_PATH.open("w", encoding="utf-8") as file:
        json.dump({"spans": spans}, file, ensure_ascii=False, indent=2)


def set_span(audio_filepath: str, song_index: int, lyric_start: int, lyric_end: int) -> None:
    spans = _load()
    spans[audio_filepath] = {
        "song_index": song_index,
        "lyric_start": int(lyric_start),
        "lyric_end": int(lyric_end),
    }
    _save(spans)


def remove_span(audio_filepath: str) -> None:
    spans = _load()
    if spans.pop(audio_filepath, None) is not None:
        _save(spans)


def rekey_span(old_path: str, new_path: str) -> None:
    """Follow a span when its segment id changes (currently ids are stable,
    but this keeps create/edit flows honest)."""
    if old_path == new_path:
        return
    spans = _load()
    if old_path in spans:
        spans[new_path] = spans.pop(old_path)
        _save(spans)


def spans_for_song(song_index: int) -> list[dict[str, Any]]:
    return [
        {"audio_filepath": path, **info}
        for path, info in _load().items()
        if int(info.get("song_index", -1)) == song_index
    ]


# --- rejected proposals ------------------------------------------------------
# Skipped suggestion keys per song id, so the review queue survives restarts.

def _load_rejected() -> dict[str, list[str]]:
    if not config.REJECTED_PATH.exists():
        return {}
    try:
        with config.REJECTED_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_rejected(rejected: dict[str, list[str]]) -> None:
    config.ensure_output_dirs()
    with config.REJECTED_PATH.open("w", encoding="utf-8") as file:
        json.dump(rejected, file, ensure_ascii=False, indent=2)


def rejected_for_song(song_id: str) -> list[str]:
    return _load_rejected().get(song_id, [])


def add_rejected(song_id: str, key: str) -> list[str]:
    rejected = _load_rejected()
    keys = rejected.setdefault(song_id, [])
    if key not in keys:
        keys.append(key)
        _save_rejected(rejected)
    return keys


def clear_rejected(song_id: str) -> None:
    rejected = _load_rejected()
    if rejected.pop(song_id, None) is not None:
        _save_rejected(rejected)
