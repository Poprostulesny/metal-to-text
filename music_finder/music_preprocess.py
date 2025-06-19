
# Import necessary libraries for data processing, file operations, and machine learning.
import json 
import os
import time
import  torch
import shutil
import config as cf
import music_utils as ut
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast

# Configure PyTorch settings for optimal performance.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)  # Dostosuj do swojego CPU


def main():
    # Set up GPU if available and optimize memory usage.
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Używam GPU: {torch.cuda.get_device_name()}")

        # Optymalizacja pamięci GPU
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.8)

    # Define the directory path where music files are stored.
    music_dir = r"C:\Users\jarek\Documents\programowanie\metal-to-text\music\\"

    # Check if a music directory exists and exit if not.
    if not os.path.isdir(music_dir):
        print("Download music files before running this script! (music_download.py)")
        quit()

    # Load music metadata from a JSON file.
    with open(music_dir+"music.json",'r', encoding='utf-8') as file:
        music_file = json.load(file)


    # Initialize empty lists to store music metadata.
    music_path = list()
    music_lyrics = list()
    music_genre = list()
    music_artist = list()

    # Extract music metadata from the loaded JSON file.
    for i in music_file:

        music_path.append(i['path'])
        music_lyrics.append(i['lyrics'])
        music_genre.append(i['genre'])
        music_artist.append(i['artist'])

    # Create a dictionary with music metadata and process it with the model.
    music_lyric_dict = {'audio_filepath': music_path, 'text': music_lyrics,"genre": music_genre, "artist": music_artist}

    with torch.cuda.amp.autocast():
        music_lyric_dict = ut.model(music_lyric_dict)

    # Create a dataset from the processed music data with audio column properly formatted.
    music_dict = list()
    for i in range(len(music_lyric_dict['text'])):
        music_dict.append({
            'audio_filepath':os.path.abspath(music_lyric_dict['audio_filepath'][i]),
            'text':music_lyric_dict['text'][i],
            'duration':ut.audio_length(music_lyric_dict['audio_filepath'][i]),
            'genre':music_lyric_dict['genre'][i],
            'artist':music_lyric_dict['artist'][i],
        })

    

    # Split dataset into train (80%) and test-valid (20%) sets.
    train_data, test_valid_data = train_test_split(music_dict, test_size=0.2, random_state=int(time.time())%137)
    # Further split test-valid into validation and test sets (50% each).
    test_data, valid_data = train_test_split(test_valid_data, test_size=0.5, random_state=int(time.time())%137)




    # Display the dataset structure for verification.
    print(music_dict)
    print(train_data)
    print(test_data)
    print(valid_data)

    # Get the output directory path from configuration.
    out_dir= cf.get_data_path()

    # Create or recreate the output directory.
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    # Save the processed dataset to disk.
    with open(out_dir + r"\test_data.jsonl", 'w', encoding="utf-8") as file:
        for i in test_data:
            file.write(json.dumps(i, ensure_ascii=True) + "\n")
    with open(out_dir + r"\train_data.jsonl", 'w', encoding="utf-8") as file:
        for i in train_data:
            file.write(json.dumps(i, ensure_ascii=True) + "\n")
    with open(out_dir + r"\valid_data.jsonl", 'w', encoding="utf-8") as file:
        for i in valid_data:
            file.write(json.dumps(i, ensure_ascii=True) + "\n")

    # Clean up GPU memory if available.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Execute the main function when the script is run directly.
if __name__ == "__main__":
    main()
