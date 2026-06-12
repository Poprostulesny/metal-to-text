"""Build augmented training clips from labeled segments, retroactively.

For every TRAIN-split segment in output/segments.jsonl this cuts the same
source_start/end window from the augmented full-song stems written by
music_preprocess (<song>-1.wav, <song>-2.wav, <song>-3.wav next to the clean
stem) and writes data/train_data_8_aug.jsonl containing the clean train rows
plus one row per available augmentation level. Valid/test splits stay clean
on purpose: augmented near-duplicates of eval clips would leak into training
metrics. The split is derived with the labeler's own seeded shuffle, so it
matches data/train_data_8.jsonl exactly.

Idempotent and retroactive: clip filenames encode the segment timing, so
re-runs only cut what is missing, re-cut segments edited in the labeler, and
prune orphaned clips. Run it any time after labeling or re-preprocessing.

Run from data-labeler/:  .venv/bin/python augment_segments.py
Requires ffmpeg/ffprobe on PATH (same as the labeler itself).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import audio, config, manifest

AUG_AUDIO_DIR = config.OUTPUT_DIR / "audio_aug"
AUG_TRAIN_MANIFEST_PATH = config.DATA_DIR / "train_data_8_aug.jsonl"
LEVELS = (1, 2, 3)


def aug_stem_path(source: Path, level: int) -> Path:
    return source.with_name(f"{source.stem}-{level}{source.suffix}")


def main() -> None:
    if not audio.ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe must be available on PATH.")

    rows = manifest.read_rows()
    if not rows:
        raise SystemExit(f"No segments in {config.MANIFEST_PATH} — label something first.")
    train, test, valid = manifest.split_rows(rows)
    print(f"{len(rows)} segments -> train {len(train)} / test {len(test)} / valid {len(valid)}")

    AUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    out_rows = list(train)
    expected: set[str] = set()
    cut = skipped = 0
    missing_stems: set[str] = set()

    for row in train:
        source = Path(row["source_audio_filepath"])
        start = float(row["source_start"])
        end = float(row["source_end"])
        for level in LEVELS:
            stem = aug_stem_path(source, level)
            if not stem.is_file():
                missing_stems.add(stem.name)
                continue
            clip_name = (f"{Path(row['audio_filepath']).stem}"
                         f"_{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}"
                         f"-{level}.wav")
            clip = AUG_AUDIO_DIR / clip_name
            expected.add(clip.name)
            if clip.exists():
                skipped += 1
            else:
                audio.cut_to_file(stem, start, end, clip)
                cut += 1
            out_rows.append({
                **row,
                "audio_filepath": str(clip.resolve()),
                "duration": round(audio.probe_duration(str(clip)), 6),
                "source_audio_filepath": str(stem.resolve()),
                "aug_level": level,
            })

    pruned = 0
    for orphan in AUG_AUDIO_DIR.glob("*.wav"):
        if orphan.name not in expected:
            orphan.unlink()
            pruned += 1

    AUG_TRAIN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUG_TRAIN_MANIFEST_PATH.open("w", encoding="utf-8") as file:
        for row in out_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Cut {cut} new clips, kept {skipped} existing, pruned {pruned} orphans.")
    print(f"Augmented train manifest: {AUG_TRAIN_MANIFEST_PATH} "
          f"({len(train)} clean + {len(out_rows) - len(train)} augmented rows)")
    if missing_stems:
        print(f"\nWARNING: {len(missing_stems)} augmented stems not found, e.g.:")
        for name in sorted(missing_stems)[:5]:
            print(f"  - {name}")
        print("Re-run music_preprocess.py to generate -1/-2/-3 stems for these songs.")
    print("\nTo train with augmentation, point model.train_ds.manifest_filepath in "
          "model/config.yaml at the augmented manifest.")


if __name__ == "__main__":
    main()
