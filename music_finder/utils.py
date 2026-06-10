import json
import os
import shutil
import project_config as pc
import logging
from progress.bar import IncrementalBar
import re

def get_text():
    music_metadata_path = pc.get_music_metadata_path()
    with open(music_metadata_path, 'r', encoding='utf-8') as file:
        music_file = json.load(file)
    text = ""

    for i in music_file:
        text = text + i['lyrics'] + " "

    text = re.sub(r"[.!?()#,;]", ' ', text)
    text =re.sub(r"\s+", ' ', text)
    text = text.lower()
    return text



