import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import project_config as pc


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def first_audio_files(music_dir: Path, limit: int) -> list[str]:
    files = [
        path
        for path in sorted(music_dir.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return [str(path) for path in files[:limit]]


def write_run_manifest(output_dir: Path, source_paths: list[str], final_paths: list[str]) -> None:
    rows = [
        {
            "source_audio_filepath": str(Path(source_path).resolve()),
            "ensemble_audio_filepath": str(Path(final_path).resolve()),
        }
        for source_path, final_path in zip(source_paths, final_paths)
    ]

    with open(output_dir / "separator_test_manifest.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=True, indent=2)


def run_separator(audio_paths: list[str], separated_dir: Path, ensemble_dir: Path, memory_threshold: float):
    import torch
    import music_utils as ut

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(1.0)
        print(f"Using GPU: {torch.cuda.get_device_name()}", flush=True)

    original_final_music_path = pc.final_music_path
    pc.final_music_path = str(ensemble_dir)
    try:
        music_dict = {"audio_filepath": audio_paths}
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            return ut.model(
                music_dict,
                memory_threshold,
                tmp_dir=str(separated_dir),
                remove_tmp_files=False,
            )
    finally:
        pc.final_music_path = original_final_music_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the production vocal separation pipeline on the first songs from music/."
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--memory-threshold", type=float, default=0.95)
    parser.add_argument("--music-dir", default=pc.get_music_path())
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "music_test"))
    args = parser.parse_args()

    music_dir = Path(args.music_dir)
    output_dir = Path(args.output_dir)
    separated_dir = output_dir / "separated_models"
    ensemble_dir = output_dir / "ensemble"

    if not music_dir.is_dir():
        print(f"Music directory does not exist: {music_dir}")
        return 1

    audio_paths = first_audio_files(music_dir, args.count)
    if not audio_paths:
        print(f"No audio files found in: {music_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    print("Separator test input files:", flush=True)
    for index, audio_path in enumerate(audio_paths, start=1):
        print(f"{index}. {audio_path}", flush=True)

    result = run_separator(audio_paths, separated_dir, ensemble_dir, args.memory_threshold)

    write_run_manifest(output_dir, audio_paths, result["audio_filepath"])

    print(f"Kept model outputs in: {separated_dir}", flush=True)
    print(f"Wrote ensemble outputs in: {ensemble_dir}", flush=True)
    print(f"Wrote manifest: {output_dir / 'separator_test_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
