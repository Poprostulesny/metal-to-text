# Import necessary libraries for data processing, file operations, and machine learning.
import json 
import os
from pathlib import Path
import sys
import time
import  torch
import music_utils as ut
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import project_config as pc




# Configure PyTorch settings for optimal performance.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(16)  # Dostosuj do swojego CPU
memory_threshold = 0.95 # Audio separator leaks memory, therefore it has to be manually cleaned up when the set memory crosses a threshold. Adjust it so that it works on your computer


def final_audio_path(source_path: str) -> str:
    return str(Path(pc.get_final_music_path()) / ut.filename(source_path))


def metadata_to_dict(rows):
    return {
        'audio_filepath': [row['path'] for row in rows],
        'text': [row['lyrics'] for row in rows],
        "genre": [row['genre'] for row in rows],
        "artist": [row['artist'] for row in rows],
    }


def build_manifest_rows(music_file):
    rows = []
    for item in music_file:
        audio_path = final_audio_path(item['path'])
        if not os.path.isfile(audio_path):
            print(f"Skipping manifest row because processed audio is missing: {audio_path}")
            continue

        rows.append({
            'audio_filepath': os.path.abspath(audio_path),
            'text': item['lyrics'],
            'duration': ut.audio_length(audio_path),
            'genre': item['genre'],
            'artist': item['artist'],
        })
    return rows


def split_manifest_rows(rows):
    if len(rows) < 3:
        return rows, [], []

    seed = int(time.time()) % 137
    train_data, test_valid_data = train_test_split(rows, test_size=0.2, random_state=seed)

    if len(test_valid_data) < 2:
        return train_data, test_valid_data, []

    test_data, valid_data = train_test_split(test_valid_data, test_size=0.5, random_state=seed)
    return train_data, test_data, valid_data

def main():
    # Set up GPU if available and optimize memory usage.
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Używam GPU: {torch.cuda.get_device_name()}")

        # Optymalizacja pamięci GPU
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(1.0)
    
    # Define the directory path where music files are stored.
    music_dir = pc.get_music_path()
    music_metadata_path = pc.get_music_metadata_path()

    # Check if a music directory exists and exit if not.
    if not os.path.isdir(music_dir):
        print("Download music files before running this script! (music_download.py)")
        quit()

    # Load music metadata from a JSON file.
    with open(music_metadata_path, 'r', encoding='utf-8') as file:
        music_file = json.load(file)


    pending_music = [item for item in music_file if not os.path.isfile(final_audio_path(item['path']))]
    skipped_count = len(music_file) - len(pending_music)
    print(f"Skipping {skipped_count} already preprocessed files.")
    print(f"Preprocessing {len(pending_music)} new files.")

    if pending_music:
        music_lyric_dict = metadata_to_dict(pending_music)
        with torch.amp.autocast("cuda", enabled=True):
            ut.model(music_lyric_dict, memory_threshold)

    music_dict = build_manifest_rows(music_file)
    train_data, test_data, valid_data = split_manifest_rows(music_dict)




    # Display the dataset structure for verification.
    print(music_dict)
    print(train_data)
    print(test_data)
    print(valid_data)

    # Get the output directory path from configuration.
    out_dir = pc.get_data_path()

    os.makedirs(out_dir, exist_ok=True)

    # Save the processed dataset to disk.
    with open(out_dir + r"/test_data_1.jsonl", 'w', encoding="utf-8") as file:
        for i in test_data:
            file.write(json.dumps(i, ensure_ascii=True) + "\n")
    with open(out_dir + r"/train_data_8.jsonl", 'w', encoding="utf-8") as file:
        for i in train_data:
            file.write(json.dumps(i, ensure_ascii=True) + "\n")
    with open(out_dir + r"/valid_data_1.jsonl", 'w', encoding="utf-8") as file:
        for i in valid_data:
            file.write(json.dumps(i, ensure_ascii=True) + "\n")

    text_corpus_dir = Path(pc.get_model_dir()) / "text_corpus"
    text_corpus_dir.mkdir(parents=True, exist_ok=True)
    with open(text_corpus_dir / "document.txt", 'w', encoding="utf-8") as file:
        line = ""
        for text in [row['text'] for row in music_dict]:
            for word in text.split():
                if len(line) + len(word) + 1 > 1000:
                    file.write(line.rstrip() + "\n")
                    line = ""
                line += word + " "
        if line:
            file.write(line.rstrip() + "\n")
    # Clean up GPU memory if available.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Execute the main function when the script is run directly.
if __name__ == "__main__":
    main()
