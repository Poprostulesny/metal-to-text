"""Central paths and constants for the labeler.

Keeping every filesystem location and tunable in one module means the rest of
the code never hard-codes a path, and the output format stays 1:1 with the
original Streamlit tool (same manifest + split filenames, same split seed).
"""
from __future__ import annotations

from pathlib import Path

# data-labeler/server/config.py -> repo root is two levels up.
LABELER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LABELER_DIR.parent

# Inputs.
MUSIC_JSON = REPO_ROOT / "music" / "music.json"
VOCAL_DIR = REPO_ROOT / "music_finder" / "final_music"

# Outputs (master manifest stays clean and 1:1 for NeMo).
OUTPUT_DIR = LABELER_DIR / "output"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
MANIFEST_PATH = OUTPUT_DIR / "segments.jsonl"
STATE_PATH = OUTPUT_DIR / "label_state.json"
VAD_CACHE_DIR = OUTPUT_DIR / "vad_cache"

# Derived split manifests (rebuilt from the master after every change).
DATA_DIR = REPO_ROOT / "data"
TRAIN_MANIFEST_PATH = DATA_DIR / "train_data_8.jsonl"
TEST_MANIFEST_PATH = DATA_DIR / "test_data_1.jsonl"
VALID_MANIFEST_PATH = DATA_DIR / "valid_data_1.jsonl"

# Audio + split tuning. Seed is fixed so the split is reproducible.
TARGET_SAMPLE_RATE = 16000
SPLIT_SEED = 137
TRAIN_FRACTION = 0.8
TEST_FRACTION = 0.1

# UI defaults.
DEFAULT_MAX_SEGMENT_SEC = 30.0
VAD_PREFETCH_AHEAD = 3

# Web assets.
WEB_DIR = LABELER_DIR / "web"


def ensure_output_dirs() -> None:
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
