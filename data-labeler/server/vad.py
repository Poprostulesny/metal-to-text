"""Voice-activity detection that proposes initial segments.

Runs the official Silero VAD model through onnxruntime (no torch), so the
labeler venv stays tiny. Results are cached on disk per song and the next few
songs are computed in a background thread, so opening a song is usually instant.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from . import config, music

MODEL_DIR = config.LABELER_DIR / "models"
MODEL_PATH = MODEL_DIR / "silero_vad.onnx"
MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)

# Window / threshold tuning (Silero defaults). Bumping VAD_VERSION invalidates
# every cached result so re-tuning is safe.
VAD_VERSION = 2
SAMPLE_RATE = config.TARGET_SAMPLE_RATE
WINDOW = 512
CONTEXT = 64  # v5 model prepends the previous 64 samples to each window
THRESHOLD = 0.5
NEG_THRESHOLD = THRESHOLD - 0.15
MIN_SPEECH_SEC = 0.25
MIN_SILENCE_SEC = 0.10
SPEECH_PAD_SEC = 0.10

_session = None
_session_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)
_scheduled: set[int] = set()


# --- model loading ---------------------------------------------------------

def _ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".onnx.part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(MODEL_PATH)
    return MODEL_PATH


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                _session = ort.InferenceSession(
                    str(_ensure_model()), sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
    return _session


# --- audio decode ----------------------------------------------------------

def _decode(path: Path) -> np.ndarray:
    """Decode any audio to mono 16 kHz float32 via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "pipe:1",
        ],
        check=True, capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


# --- inference -------------------------------------------------------------

def _speech_probs(samples: np.ndarray) -> np.ndarray:
    session = _get_session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    context = np.zeros(CONTEXT, dtype=np.float32)
    probs = []
    for start in range(0, len(samples), WINDOW):
        chunk = samples[start:start + WINDOW]
        if len(chunk) < WINDOW:
            chunk = np.pad(chunk, (0, WINDOW - len(chunk)))
        window = np.concatenate([context, chunk])
        out, state = session.run(
            None, {"input": window[np.newaxis, :], "state": state, "sr": sr}
        )
        probs.append(float(out[0][0]))
        context = chunk[-CONTEXT:]
    return np.array(probs, dtype=np.float32)


def _probs_to_regions(probs: np.ndarray, total_samples: int) -> list[dict[str, float]]:
    min_speech = MIN_SPEECH_SEC * SAMPLE_RATE
    min_silence = MIN_SILENCE_SEC * SAMPLE_RATE
    pad = SPEECH_PAD_SEC * SAMPLE_RATE

    speeches: list[dict[str, float]] = []
    current: dict[str, float] = {}
    triggered = False
    temp_end = 0.0

    for i, prob in enumerate(probs):
        sample = i * WINDOW
        if prob >= THRESHOLD and temp_end:
            temp_end = 0.0
        if prob >= THRESHOLD and not triggered:
            triggered = True
            current = {"start": sample}
            continue
        if prob < NEG_THRESHOLD and triggered:
            if not temp_end:
                temp_end = sample
            if sample - temp_end < min_silence:
                continue
            current["end"] = temp_end
            if current["end"] - current["start"] > min_speech:
                speeches.append(current)
            current, temp_end, triggered = {}, 0.0, False

    if current and (total_samples - current["start"]) > min_speech:
        current["end"] = total_samples
        speeches.append(current)

    # Symmetric padding, clamped so neighbours don't overlap.
    for i, sp in enumerate(speeches):
        sp["start"] = max(0.0, sp["start"] - pad)
        if i + 1 < len(speeches):
            gap = speeches[i + 1]["start"] - sp["end"]
            grow = min(pad, gap / 2)
            sp["end"] += grow
        else:
            sp["end"] = min(total_samples, sp["end"] + pad)

    return [
        {"start": round(sp["start"] / SAMPLE_RATE, 3),
         "end": round(sp["end"] / SAMPLE_RATE, 3)}
        for sp in speeches
    ]


# --- caching + public api --------------------------------------------------

def _cache_path(song: dict[str, Any]) -> Path:
    return config.VAD_CACHE_DIR / f"{music.song_id(song)}.json"


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{stat.st_size}:{int(stat.st_mtime)}:{VAD_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()


def regions_for_song(song_index: int) -> list[dict[str, float]]:
    song = music.get_song(song_index)
    source = music.vocal_stem_path(song)
    if not source.exists():
        return []

    config.ensure_output_dirs()
    cache, fp = _cache_path(song), _fingerprint(source)
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fp:
                return cached["regions"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    samples = _decode(source)
    regions = _probs_to_regions(_speech_probs(samples), len(samples))
    cache.write_text(
        json.dumps({"fingerprint": fp, "regions": regions}, ensure_ascii=False),
        encoding="utf-8",
    )
    return regions


def prefetch(after_index: int, ahead: int = config.VAD_PREFETCH_AHEAD) -> None:
    """Warm the cache for the next songs in the background."""
    total = music.song_count()
    for index in range(after_index + 1, min(after_index + 1 + ahead, total)):
        if index in _scheduled:
            continue
        _scheduled.add(index)
        _pool.submit(_prefetch_one, index)


def _prefetch_one(index: int) -> None:
    try:
        regions_for_song(index)
    except Exception:
        pass
    finally:
        _scheduled.discard(index)
