"""Check synced (LRC) lyrics coverage for songs in music.json, per provider.

Queries each community LRC database separately (Musixmatch, LRCLIB, NetEase,
Megalobiz) via the `syncedlyrics` package and reports which providers have a
synced match for each song. The best available LRC (by provider priority) is
saved to music/lrc/ for later reuse (e.g. as region proposals in the data
labeler). Per-song results are cached in music/lrc/coverage.json, so re-runs
only query songs that previously errored or are new.

Run with: .venv-data/bin/python music_finder/check_synced_lyrics.py
Requires: pip install syncedlyrics
"""

import json
import re
import sys
import time
from pathlib import Path

import syncedlyrics

sys.path.append(str(Path(__file__).resolve().parent.parent))
from project_config import get_music_metadata_path, get_music_path

# Priority order: first provider with a hit is the one whose LRC gets saved.
# Genius is intentionally absent — it only serves plain (unsynced) lyrics.
PROVIDERS = ["Musixmatch", "Lrclib", "NetEase", "Megalobiz"]

# A line is considered synced if it contains an [mm:ss.xx] timestamp.
TIMESTAMP_RE = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")


def main() -> None:
    with open(get_music_metadata_path(), encoding="utf-8") as f:
        songs = json.load(f)

    lrc_dir = Path(get_music_path()) / "lrc"
    lrc_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = lrc_dir / "coverage.json"

    coverage = {}
    if coverage_path.exists():
        with open(coverage_path, encoding="utf-8") as f:
            coverage = json.load(f)

    for i, song in enumerate(songs):
        # Filenames follow the "Artist - Title.wav" spotdl convention.
        track_name = Path(song["path"]).stem

        cached = coverage.get(track_name)
        if cached is not None and not cached.get("error"):
            print(f"[{i + 1}/{len(songs)}] CACHED {track_name}: "
                  f"{', '.join(cached['providers']) or 'no match'}")
            continue

        hits = {}
        error = False
        for provider in PROVIDERS:
            try:
                lrc = syncedlyrics.search(track_name, providers=[provider])
            except Exception as e:
                print(f"[{i + 1}/{len(songs)}] ERROR  {track_name} "
                      f"({provider}: {e})")
                error = True
                continue
            if lrc and TIMESTAMP_RE.search(lrc):
                hits[provider] = lrc
            # Be gentle with the providers, they are scraped community services.
            time.sleep(0.3)

        if hits:
            best_provider = next(p for p in PROVIDERS if p in hits)
            (lrc_dir / f"{track_name}.lrc").write_text(
                hits[best_provider], encoding="utf-8")

        coverage[track_name] = {
            "providers": [p for p in PROVIDERS if p in hits],
            "error": error,
        }
        with open(coverage_path, "w", encoding="utf-8") as f:
            json.dump(coverage, f, ensure_ascii=False, indent=2)

        status = ", ".join(coverage[track_name]["providers"]) or "no match"
        print(f"[{i + 1}/{len(songs)}] {track_name}: {status}")

    total = len(songs)
    checked = [coverage[Path(s["path"]).stem] for s in songs
               if Path(s["path"]).stem in coverage]
    have = [c for c in checked if c["providers"]]
    print("\n=== Synced lyrics coverage ===")
    print(f"Any provider: {len(have)}/{total} ({100 * len(have) / total:.0f}%)")
    for provider in PROVIDERS:
        count = sum(1 for c in checked if provider in c["providers"])
        print(f"  {provider:<11} {count}/{total}")
    errors = sum(1 for c in checked if c["error"])
    if errors:
        print(f"Errors: {errors} songs had provider failures (re-run to retry)")
    print(f"LRC files saved in: {lrc_dir}")

    missing = [Path(s["path"]).stem for s in songs
               if not coverage.get(Path(s["path"]).stem, {}).get("providers")]
    if missing:
        print("\nSongs without synced lyrics:")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
