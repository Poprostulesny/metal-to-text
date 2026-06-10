# This file is needed due to memory leaks in htdemucs audio_separator cuda backend.
import argparse
import json
import sys
import traceback
from pathlib import Path

from audio_separator.separator import Separator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_results(output_json: str, results: list[dict]) -> None:
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=True, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunk-start", type=int, required=True)
    parser.add_argument("--chunk-total", type=int, required=True)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as file:
        jobs = json.load(file)

    sep = Separator(
        output_dir=args.output_dir,
        output_format="WAV",
        output_single_stem="Vocals",
        use_autocast=True,
        demucs_params={
            "segment_size": 30,
            "shifts": 0,
            "overlap": 0.1,
            "segments_enabled": True,
        },
    )
    sep.load_model(args.model)

    results = []
    failed = False
    for chunk_index, job in enumerate(jobs, start=1):
        input_path = job["input"]
        global_index = job["index"]
        total_songs = job["total"]
        name = Path(input_path).name
        print(
            f"{args.model}: processing song {global_index}/{total_songs} "
            f"(chunk {args.chunk_start}/{args.chunk_total}, item {chunk_index}/{len(jobs)}): {name}",
            flush=True,
        )

        try:
            outputs = sep.separate(input_path)
            results.append(
                {
                    "index": global_index,
                    "input": input_path,
                    "outputs": outputs,
                    "ok": True,
                    "error": None,
                }
            )
        except Exception:
            failed = True
            results.append(
                {
                    "index": global_index,
                    "input": input_path,
                    "outputs": [],
                    "ok": False,
                    "error": traceback.format_exc(),
                }
            )

        write_results(args.output_json, results)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
