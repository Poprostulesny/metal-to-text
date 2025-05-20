
import json 
from datasets import  DatasetDict, Dataset, Audio
import os
import config as cf
import time
import shutil
def main():
    music_dir = r"C:\Users\jarek\Documents\programowanie\metal-to-text\music\\"
    if not os.path.isdir(music_dir):
        print("Download music files before running this script! (music_download.py)")
        quit()

    with open(music_dir+"music.json",'r', encoding='utf-8') as file:
        music_file = json.load(file)

    music_dict = DatasetDict()

    music_path = list()
    music_lyrics = list()
    for i in music_file:
        music_path.append(i['path'])
        music_lyrics.append(i['lyrics'])



    # splitting the dataset
    music_dict = Dataset.from_dict({"lyrics":music_lyrics,"audio": music_path}).cast_column("audio", Audio())
    music_dict = music_dict.shuffle(seed=int(time.time())%1057 )

    train_testvalid = music_dict.train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

    music_dict = DatasetDict({
        "train": train_testvalid['train'],
        "valid": test_valid['train'],
        "test": test_valid['test']

    })
    print(music_dict)
    # saving the dataset to the file
    out_dir= cf.get_data_path()


    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    print("Saving data")

    music_dict.save_to_disk(out_dir)

if __name__ == "__main__":
    main()