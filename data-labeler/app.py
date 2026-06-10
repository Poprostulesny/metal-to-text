from __future__ import annotations

import json
import re
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
MUSIC_JSON = REPO_ROOT / "music" / "music.json"
VOCAL_DIR = REPO_ROOT / "music_finder" / "final_music"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
MANIFEST_PATH = OUTPUT_DIR / "segments.jsonl"
DATA_DIR = REPO_ROOT / "data"
TRAIN_MANIFEST_PATH = DATA_DIR / "train_data_8.jsonl"
TEST_MANIFEST_PATH = DATA_DIR / "test_data_1.jsonl"
VALID_MANIFEST_PATH = DATA_DIR / "valid_data_1.jsonl"
TARGET_SAMPLE_RATE = 16000
SPLIT_SEED = 137


@dataclass(frozen=True)
class SegmentDraft:
    start_sec: float
    end_sec: float
    text: str


def ensure_output_dirs() -> None:
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def load_music_metadata() -> list[dict[str, Any]]:
    if not MUSIC_JSON.exists():
        raise FileNotFoundError(f"Missing metadata file: {MUSIC_JSON}")
    with MUSIC_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {MUSIC_JSON}")
    return data


@st.cache_data(show_spinner=False)
def audio_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


@st.cache_data(show_spinner=False)
def render_segment_preview(path: str, start_sec: float, end_sec: float) -> bytes:
    return cut_segment_to_bytes(Path(path), start_sec, end_sec)


def cut_segment_to_bytes(path: Path, start_sec: float, end_sec: float) -> bytes:
    duration = max(0.0, end_sec - start_sec)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-f",
            "wav",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout


def cut_segment_to_file(path: Path, start_sec: float, end_sec: float, output_path: Path) -> None:
    duration = max(0.0, end_sec - start_sec)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            str(output_path),
        ],
        check=True,
    )


def vocal_stem_path(song: dict[str, Any]) -> Path:
    raw_path = Path(str(song.get("path", "")))
    return VOCAL_DIR / raw_path.name


def song_label(song: dict[str, Any], index: int) -> str:
    raw_path = Path(str(song.get("path", "")))
    artist = str(song.get("artist") or "Unknown artist")
    title = raw_path.stem or f"Song {index + 1}"
    return f"{index + 1:03d}. {artist} - {title}"


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or "segment"


def existing_manifest_rows() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_manifest_rows(rows: list[dict[str, Any]]) -> None:
    ensure_output_dirs()
    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    shuffled = list(rows)
    random.Random(SPLIT_SEED).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * 0.8)
    test_count = int(total * 0.1)

    if total > 0 and train_count == 0:
        train_count = 1
    if total >= 3 and test_count == 0:
        test_count = 1

    valid_count = total - train_count - test_count
    if total >= 3 and valid_count == 0:
        valid_count = 1
        train_count = max(1, train_count - 1)

    train_rows = shuffled[:train_count]
    test_rows = shuffled[train_count : train_count + test_count]
    valid_rows = shuffled[train_count + test_count :]
    return train_rows, test_rows, valid_rows


def export_data_splits() -> tuple[int, int, int]:
    rows = existing_manifest_rows()
    train_rows, test_rows, valid_rows = split_rows(rows)
    write_jsonl(TRAIN_MANIFEST_PATH, train_rows)
    write_jsonl(TEST_MANIFEST_PATH, test_rows)
    write_jsonl(VALID_MANIFEST_PATH, valid_rows)
    return len(train_rows), len(test_rows), len(valid_rows)


def rows_for_source(source_audio_path: Path) -> list[dict[str, Any]]:
    source = str(source_audio_path.resolve())
    return [
        row
        for row in existing_manifest_rows()
        if row.get("source_audio_filepath") == source
    ]


def undo_last_segment_for_source(source_audio_path: Path) -> tuple[bool, str]:
    rows = existing_manifest_rows()
    source = str(source_audio_path.resolve())
    target_index: int | None = None

    for index in range(len(rows) - 1, -1, -1):
        if rows[index].get("source_audio_filepath") == source:
            target_index = index
            break

    if target_index is None:
        return False, "No saved segment to undo for this song."

    row = rows.pop(target_index)
    audio_path = Path(str(row.get("audio_filepath", "")))
    write_manifest_rows(rows)
    export_data_splits()

    deleted_audio = False
    try:
        resolved_audio = audio_path.resolve()
        resolved_output = AUDIO_OUTPUT_DIR.resolve()
        if resolved_audio.is_file() and resolved_output in resolved_audio.parents:
            resolved_audio.unlink()
            deleted_audio = True
    except OSError:
        deleted_audio = False

    filename = audio_path.name or "segment"
    if deleted_audio:
        return True, f"Removed {filename} and its manifest row."
    return True, f"Removed manifest row for {filename}. Audio file was not deleted."


def validate_segment(draft: SegmentDraft, source_path: Path, source_duration: float) -> list[str]:
    errors: list[str] = []
    if not source_path.exists():
        errors.append(f"Source audio does not exist: {source_path}")
    if draft.start_sec < 0:
        errors.append("Start must be >= 0.")
    if draft.end_sec <= draft.start_sec:
        errors.append("End must be greater than start.")
    if draft.end_sec > source_duration:
        errors.append(f"End must be <= audio duration ({source_duration:.2f}s).")
    if not draft.text.strip():
        errors.append("Segment text cannot be empty.")
    return errors


def next_segment_path(song: dict[str, Any], draft: SegmentDraft) -> Path:
    raw_name = Path(str(song.get("path", ""))).stem
    song_id = safe_slug(raw_name)
    start_ms = int(round(draft.start_sec * 1000))
    end_ms = int(round(draft.end_sec * 1000))
    base = f"{song_id}_{start_ms:09d}_{end_ms:09d}"
    counter = 1
    while True:
        path = AUDIO_OUTPUT_DIR / f"{base}_{counter:03d}.wav"
        if not path.exists():
            return path
        counter += 1


def save_segment(song: dict[str, Any], source_path: Path, source_index: int, draft: SegmentDraft) -> dict[str, Any]:
    ensure_output_dirs()
    segment_path = next_segment_path(song, draft)
    cut_segment_to_file(source_path, draft.start_sec, draft.end_sec, segment_path)

    duration = audio_duration(str(segment_path))
    row = {
        "audio_filepath": str(segment_path.resolve()),
        "duration": round(duration, 6),
        "text": draft.text.strip(),
        "artist": song.get("artist") or "",
        "genre": song.get("genre") or [],
        "source_audio_filepath": str(source_path.resolve()),
        "source_start": round(float(draft.start_sec), 6),
        "source_end": round(float(draft.end_sec), 6),
        "source_index": source_index,
    }

    with MANIFEST_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    export_data_splits()
    return row


def set_song_index(index: int, max_index: int) -> None:
    st.session_state.song_index = max(0, min(index, max_index))
    st.session_state.start_sec = 0.0
    st.session_state.end_sec = 30.0
    st.session_state.segment_text = ""
    refresh_time_inputs()


def refresh_time_inputs() -> None:
    st.session_state.time_input_version = int(st.session_state.get("time_input_version", 0)) + 1


def render_sidebar(songs: list[dict[str, Any]]) -> int:
    max_index = len(songs) - 1
    if "song_index" not in st.session_state:
        st.session_state.song_index = 0
    st.session_state.song_index = max(0, min(int(st.session_state.song_index), max_index))

    labels = [song_label(song, index) for index, song in enumerate(songs)]
    selected_index = st.sidebar.selectbox(
        "Song",
        list(range(len(songs))),
        index=int(st.session_state.song_index),
        format_func=lambda index: labels[index],
    )
    if int(selected_index) != int(st.session_state.song_index):
        set_song_index(int(selected_index), max_index)
        st.rerun()

    if st.sidebar.button("Next song", use_container_width=True):
        if int(st.session_state.song_index) >= max_index:
            st.sidebar.info("Already at the last song.")
        else:
            set_song_index(int(st.session_state.song_index) + 1, max_index)
            st.rerun()

    st.sidebar.caption(f"{int(st.session_state.song_index) + 1} / {len(songs)}")
    if st.sidebar.button("Export train/test/valid", use_container_width=True):
        train_count, test_count, valid_count = export_data_splits()
        st.sidebar.success(
            f"Exported {train_count} train, {test_count} test, {valid_count} valid."
        )
    return int(st.session_state.song_index)


def split_time(total_seconds: float) -> tuple[int, float]:
    minutes = int(max(0.0, total_seconds) // 60)
    seconds = min(59.999, max(0.0, total_seconds) - minutes * 60)
    return minutes, seconds


def combine_time(minutes: int, seconds: float) -> float:
    return float(minutes) * 60.0 + float(seconds)


def render_existing_segments(source_path: Path) -> None:
    rows = rows_for_source(source_path)
    st.subheader("Saved segments for this song")
    if not rows:
        st.caption("No segments saved for this source yet.")
        return

    if st.button("Undo last segment for this song", use_container_width=True):
        ok, message = undo_last_segment_for_source(source_path)
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.info(message)

    table_rows = [
        {
            "start": row.get("source_start"),
            "end": row.get("source_end"),
            "duration": row.get("duration"),
            "text": row.get("text"),
            "audio": Path(str(row.get("audio_filepath", ""))).name,
        }
        for row in rows
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_split_status() -> None:
    rows = existing_manifest_rows()
    train_rows, test_rows, valid_rows = split_rows(rows)
    st.caption(
        "Current split: "
        f"{len(train_rows)} train / {len(test_rows)} test / {len(valid_rows)} valid. "
        "Splits are auto-exported after save and undo."
    )


def run_app() -> None:
    st.set_page_config(page_title="Metal-to-text labeler", layout="wide")
    st.title("Metal-to-text data labeler")

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        st.error("ffmpeg and ffprobe must be available on PATH.")
        return

    try:
        songs = load_music_metadata()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(str(error))
        return

    if not songs:
        st.warning(f"No songs found in {MUSIC_JSON}")
        return

    index = render_sidebar(songs)
    song = songs[index]
    source_path = vocal_stem_path(song)

    st.caption(f"Metadata: `{MUSIC_JSON}`")
    render_split_status()
    st.header(song_label(song, index))
    st.write(f"Artist: `{song.get('artist') or 'Unknown artist'}`")
    st.write(f"Vocal source: `{source_path}`")

    if not source_path.exists():
        st.error(f"Vocal stem not found: {source_path}")
        st.info("Expected a vocal-only WAV with the same filename as the raw song.")
        st.subheader("Lyrics")
        st.text_area("Full lyrics", str(song.get("lyrics") or ""), height=320, disabled=True)
        return

    try:
        duration = audio_duration(str(source_path))
    except (subprocess.CalledProcessError, ValueError) as error:
        st.error(f"Could not read source audio: {error}")
        return

    if "start_sec" not in st.session_state:
        st.session_state.start_sec = 0.0
    if "end_sec" not in st.session_state:
        st.session_state.end_sec = min(30.0, duration)
    if "segment_text" not in st.session_state:
        st.session_state.segment_text = ""
    if "time_input_version" not in st.session_state:
        st.session_state.time_input_version = 0

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Audio")
        st.caption(f"Duration: {duration:.2f}s")
        st.audio(str(source_path), format="audio/wav")

        start_default_min, start_default_sec = split_time(float(st.session_state.start_sec))
        end_default_min, end_default_sec = split_time(min(float(st.session_state.end_sec), duration))
        max_minutes = int(duration // 60) + 1
        time_key = f"{index}_{int(st.session_state.time_input_version)}"

        with st.form("segment_form", clear_on_submit=False):
            start_cols = st.columns(2)
            with start_cols[0]:
                start_min = st.number_input(
                    "Start minutes",
                    min_value=0,
                    max_value=max_minutes,
                    value=start_default_min,
                    step=1,
                    key=f"start_min_{time_key}",
                )
            with start_cols[1]:
                start_second_part = st.number_input(
                    "Start seconds",
                    min_value=0.0,
                    max_value=59.999,
                    value=start_default_sec,
                    step=0.1,
                    format="%.3f",
                    key=f"start_sec_{time_key}",
                )

            end_cols = st.columns(2)
            with end_cols[0]:
                end_min = st.number_input(
                    "End minutes",
                    min_value=0,
                    max_value=max_minutes,
                    value=end_default_min,
                    step=1,
                    key=f"end_min_{time_key}",
                )
            with end_cols[1]:
                end_second_part = st.number_input(
                    "End seconds",
                    min_value=0.0,
                    max_value=59.999,
                    value=end_default_sec,
                    step=0.1,
                    format="%.3f",
                    key=f"end_sec_{time_key}",
                )
            text = st.text_area(
                "Segment text",
                value=str(st.session_state.segment_text),
                height=160,
            )
            preview_clicked, save_clicked = st.columns(2)
            preview = preview_clicked.form_submit_button("Preview segment", use_container_width=True)
            save = save_clicked.form_submit_button("Save segment", use_container_width=True)

        start_sec = combine_time(start_min, start_second_part)
        end_sec = combine_time(end_min, end_second_part)
        draft = SegmentDraft(start_sec=start_sec, end_sec=end_sec, text=text)
        st.session_state.start_sec = start_sec
        st.session_state.end_sec = end_sec
        st.session_state.segment_text = text

        errors = validate_segment(draft, source_path, duration)
        if preview:
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    st.audio(
                        render_segment_preview(str(source_path), draft.start_sec, draft.end_sec),
                        format="audio/wav",
                    )
                except subprocess.CalledProcessError as error:
                    st.error(f"ffmpeg failed while rendering preview: {error}")

        if save:
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    row = save_segment(song, source_path, index, draft)
                    st.success(f"Saved {Path(row['audio_filepath']).name}")
                    st.session_state.start_sec = draft.end_sec
                    st.session_state.end_sec = min(draft.end_sec + 30.0, duration)
                    st.session_state.segment_text = ""
                    refresh_time_inputs()
                    st.rerun()
                except subprocess.CalledProcessError as error:
                    st.error(f"ffmpeg failed while saving segment: {error}")

    with right:
        st.subheader("Lyrics")
        st.text_area("Full lyrics", str(song.get("lyrics") or ""), height=420, disabled=True)
        render_existing_segments(source_path)


def main() -> None:
    try:
        run_app()
    except Exception:
        st.error("Data labeler crashed while rendering.")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
