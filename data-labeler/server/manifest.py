"""The master manifest and derived train/test/valid splits.

This is the part of the original tool that was already good and is kept 1:1:
``output/segments.jsonl`` is the single source of truth, and the ``data/``
split manifests are *rebuilt* from it (seeded shuffle, 80/10/10) after every
mutation rather than being appended to or recreated ad hoc.

A segment's ``audio_filepath`` is its stable id: clip filenames never collide,
so we never store an extra id field and the manifest stays NeMo-clean.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from . import audio, config, music


# --- low level jsonl io ----------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows() -> list[dict[str, Any]]:
    return _read_jsonl(config.MANIFEST_PATH)


# --- split derivation ------------------------------------------------------

def split_rows(rows: list[dict[str, Any]]) -> tuple[list, list, list]:
    shuffled = list(rows)
    random.Random(config.SPLIT_SEED).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * config.TRAIN_FRACTION)
    test_count = int(total * config.TEST_FRACTION)

    if total > 0 and train_count == 0:
        train_count = 1
    if total >= 3 and test_count == 0:
        test_count = 1

    valid_count = total - train_count - test_count
    if total >= 3 and valid_count == 0:
        valid_count = 1
        train_count = max(1, train_count - 1)

    train = shuffled[:train_count]
    test = shuffled[train_count: train_count + test_count]
    valid = shuffled[train_count + test_count:]
    return train, test, valid


def export_splits() -> dict[str, int]:
    train, test, valid = split_rows(read_rows())
    _write_jsonl(config.TRAIN_MANIFEST_PATH, train)
    _write_jsonl(config.TEST_MANIFEST_PATH, test)
    _write_jsonl(config.VALID_MANIFEST_PATH, valid)
    return {"train": len(train), "test": len(test), "valid": len(valid)}


# --- segment crud ----------------------------------------------------------

def _next_clip_path(song: dict[str, Any], start_sec: float, end_sec: float) -> Path:
    base = f"{music.song_id(song)}_{int(round(start_sec * 1000)):09d}_{int(round(end_sec * 1000)):09d}"
    counter = 1
    while True:
        path = config.AUDIO_OUTPUT_DIR / f"{base}_{counter:03d}.wav"
        if not path.exists():
            return path
        counter += 1


def _build_row(song: dict[str, Any], source: Path, index: int,
               clip: Path, start_sec: float, end_sec: float, text: str) -> dict[str, Any]:
    return {
        "audio_filepath": str(clip.resolve()),
        "duration": round(audio.probe_duration(str(clip)), 6),
        "text": text.strip(),
        "artist": song.get("artist") or "",
        "genre": song.get("genre") or [],
        "source_audio_filepath": str(source.resolve()),
        "source_start": round(float(start_sec), 6),
        "source_end": round(float(end_sec), 6),
        "source_index": index,
    }


def add_segment(song_index: int, start_sec: float, end_sec: float, text: str) -> dict[str, Any]:
    config.ensure_output_dirs()
    song = music.get_song(song_index)
    source = music.vocal_stem_path(song)
    clip = _next_clip_path(song, start_sec, end_sec)
    audio.cut_to_file(source, start_sec, end_sec, clip)
    row = _build_row(song, source, song_index, clip, start_sec, end_sec, text)

    rows = read_rows()
    rows.append(row)
    _write_jsonl(config.MANIFEST_PATH, rows)
    export_splits()
    return row


def update_segment(audio_filepath: str, start_sec: float, end_sec: float, text: str) -> dict[str, Any]:
    rows = read_rows()
    target = _find(rows, audio_filepath)
    song = music.get_song(int(target.get("source_index", 0)))
    source = music.vocal_stem_path(song)

    # Re-cut in place; the clip keeps its filename so the id stays stable.
    clip = Path(audio_filepath)
    audio.cut_to_file(source, start_sec, end_sec, clip)
    target.update(_build_row(song, source, int(target.get("source_index", 0)),
                             clip, start_sec, end_sec, text))
    _write_jsonl(config.MANIFEST_PATH, rows)
    export_splits()
    return target


def delete_segment(audio_filepath: str) -> dict[str, Any]:
    rows = read_rows()
    target = _find(rows, audio_filepath)
    rows = [r for r in rows if r.get("audio_filepath") != audio_filepath]
    _write_jsonl(config.MANIFEST_PATH, rows)
    export_splits()
    _unlink_clip(Path(audio_filepath))
    return target


def rows_for_song(song_index: int) -> list[dict[str, Any]]:
    return [r for r in read_rows() if int(r.get("source_index", -1)) == song_index]


def _find(rows: list[dict[str, Any]], audio_filepath: str) -> dict[str, Any]:
    for row in rows:
        if row.get("audio_filepath") == audio_filepath:
            return row
    raise KeyError(f"No segment with audio_filepath={audio_filepath}")


def _unlink_clip(clip: Path) -> None:
    """Delete a clip only if it lives inside our managed audio output dir."""
    try:
        resolved = clip.resolve()
        if resolved.is_file() and config.AUDIO_OUTPUT_DIR.resolve() in resolved.parents:
            resolved.unlink()
    except OSError:
        pass
