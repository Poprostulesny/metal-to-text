"""ffmpeg/ffprobe helpers.

Audio probing and cutting stay in system ffmpeg so the labeler venv stays
small. Every saved clip is mono 16 kHz WAV, matching the original tool.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from . import config


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_duration(path: str) -> float:
    """Probe duration fresh. Use for clips, which get re-cut in place."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


@lru_cache(maxsize=256)
def duration_sec(path: str) -> float:
    """Cached duration. Only for immutable source stems."""
    return probe_duration(path)


def _ffmpeg_cut_args(path: Path, start_sec: float, end_sec: float) -> list[str]:
    duration = max(0.0, end_sec - start_sec)
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}",
        "-t", f"{duration:.6f}",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(config.TARGET_SAMPLE_RATE),
    ]


def cut_to_bytes(path: Path, start_sec: float, end_sec: float) -> bytes:
    """Render a region to an in-memory WAV (used for previews)."""
    result = subprocess.run(
        [*_ffmpeg_cut_args(path, start_sec, end_sec), "-f", "wav", "pipe:1"],
        check=True, capture_output=True,
    )
    return result.stdout


def cut_to_file(path: Path, start_sec: float, end_sec: float, out_path: Path) -> None:
    subprocess.run(
        [*_ffmpeg_cut_args(path, start_sec, end_sec), "-y", str(out_path)],
        check=True, capture_output=True,
    )
