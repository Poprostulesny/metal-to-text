import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import project_config as pc


MAX_LINE_LENGTH = 1000


def write_wrapped_text(texts: list[str], output_path: Path, max_line_length: int = MAX_LINE_LENGTH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        line = ""
        for text in texts:
            for word in text.split():
                if len(line) + len(word) + 1 > max_line_length:
                    file.write(line.rstrip() + "\n")
                    line = ""
                line += word + " "

        if line:
            file.write(line.rstrip() + "\n")


def main() -> None:
    music_metadata_path = Path(pc.get_music_metadata_path())
    output_path = Path(pc.get_model_dir()) / "text_corpus" / "document.txt"

    with open(music_metadata_path, "r", encoding="utf-8") as file:
        music_items = json.load(file)

    lyrics = [
        item.get("lyrics", "").strip()
        for item in music_items
        if item.get("lyrics", "").strip()
    ]

    write_wrapped_text(lyrics, output_path)
    print(f"Wrote {len(lyrics)} lyric entries to {output_path}", flush=True)


if __name__ == "__main__":
    main()
