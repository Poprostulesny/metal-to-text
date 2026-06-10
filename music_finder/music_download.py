from spotdl import Spotdl
import os
import json
import sys
from pathlib import Path
from spotdl.types.options import DownloaderOptions
from spotdl.types.playlist import Playlist
import music_utils as ut
import logging
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import project_config as pc

def load_existing_metadata(music_metadata_path):
    if not os.path.exists(music_metadata_path):
        return []

    with open(music_metadata_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {music_metadata_path}")

    return data


def metadata_keys(entries):
    keys = set()
    for entry in entries:
        path = entry.get("path")
        if not path:
            continue
        keys.add(os.path.abspath(path))
        keys.add(os.path.basename(path))
    return keys


def main():
    url = pc.get_music_url()
    model = pc.get_model()
    location_dir = pc.get_music_path()
    ollama_url = pc.get_ollama_url()
    music_metadata_path = pc.get_music_metadata_path()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    os.makedirs(location_dir, exist_ok=True)
    songs_new = load_existing_metadata(music_metadata_path)
    existing_song_keys = metadata_keys(songs_new)
    print(f"Loaded {len(songs_new)} existing metadata rows from {music_metadata_path}")

    downloader_options: DownloaderOptions = {
        # where and how files should be saved
        "audio_providers": ["youtube-music"],
        # spotdl has some problems with getting lyrics from other providers, therefore this line is left to uncomment in the case they fix it
        # "lyrics_providers": ["genius", "azlyrics", "musixmatch", "synced"],
        "lyrics_providers": ["synced"],
        "genius_token": pc.get_genius_token(),
        "playlist_numbering": False,
        "playlist_retain_track_cover": False,
        "scan_for_songs": False,
        "m3u": None,
        "output": os.path.join(location_dir, "{artists} - {title}.{output-ext}"),
        "overwrite": "skip",
        "search_query": None,
        "ffmpeg": "ffmpeg",
        "bitrate": "256k",
        "ffmpeg_args": None,
        "format": "wav",
        "save_file": "spotdl_cache.txt",
        "filter_results": True,
        "album_type": None,
        "threads": 16,
        "cookie_file": None,
        "restrict": None,
        "print_errors": False,
        "sponsor_block": False,
        "preload": False,
        "archive": None,
        "load_config": True,
        "log_level": "INFO",
        "simple_tui": False,
        "fetch_albums": False,
        "id3_separator": "/",
        "ytm_data": False,
        "add_unavailable": False,
        "generate_lrc": False,
        "force_update_metadata": False,
        "only_verified_results": False,
        "sync_without_deleting": False,
        "max_filename_length": None,
        "yt_dlp_args": None,
        "detect_formats": None,
        "save_errors": "spotdl_errors.txt",
        "ignore_albums": None,
        "proxy": None,
        "skip_explicit": False,
        "log_format": None ,
        "redownload": False,
        "skip_album_art": True,
        "create_skip_file": False,
        "respect_skip_file": False,
        "sync_remove_lrc": False,
    }



    message = 'you get this lyric string downloaded from a site, output to me only the lyrics without any special characters and anything else not being sang, no other words should come out of you then the output string remember to add spaces or interpunction marks where needed, delete every enter and quotation marks.'
    spotdl = Spotdl(client_id='d2bc39f6f3ba4f86ba702ce43d9611e7', client_secret='349bdc3f646e4dca83efca5db4e2ff09',downloader_settings=downloader_options)
    
    
    # meta, songs = Playlist.get_metadata(url)
    print("Fetching playlist...")
    playlist =  Playlist.from_url(url, fetch_songs=False)
    songs = playlist.songs
    
    # optimizing so that spotdl isnt making thousends of unndeded api calls for the metadata
    for song in songs:
        if song.genres is None:
            song.genres = []
        if song.disc_count is None:
            song.disc_count = 1
        if song.tracks_count is None:
            song.tracks_count = 1
        if song.track_number is None:
            song.track_number = song.list_position or 1
        if song.album_id is None:
            song.album_id = ""
        if song.album_artist is None:
            song.album_artist = song.artist
        if song.publisher is None:
            song.publisher = ""
        if song.date is None:
            song.date = ""
        if song.isrc is None:
            song.isrc = ""
        if song.album_name is None:
            song.album_name = ""
        if song.url is None:
            song.url = ""

            
    
    print("Starting download")
    songs = spotdl.download_songs(songs)

    songs_bad =list[str]()
    songs_good = {entry.get("path") for entry in songs_new if entry.get("path")}
    if os.name == 'nt':
        os.system('cls')
    else:
        print("\033[2J\033[H", end="", flush=True)
    # print(songs[1])
    for song, path in tqdm(
        songs,
        desc='Processing downloaded songs',
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    ):
        if path is not None:
            path_key = os.path.abspath(str(path))
            filename_key = os.path.basename(str(path))
            if path_key in existing_song_keys or filename_key in existing_song_keys:
                songs_good.add(str(path))
                continue

        if song.lyrics is not None and path is not None:
            song.lyrics = ut.sanitized_lyrics(song.lyrics, ollama_url, model, message)
            songs_new.append({
                "path":   str(path),
                "lyrics": song.lyrics,
                "artist":song.artist,
                "genre":song.genres,
            })
            songs_good.add(str(path)) 
            existing_song_keys.add(os.path.abspath(str(path)))
            existing_song_keys.add(os.path.basename(str(path)))
        elif(path is not None):
                songs_bad.append(str(path))


    if len(songs_bad)!=0:
        for i in songs_bad:
            if i is None or i == "None":
                continue
            if i not in songs_good:
                try:
                    os.remove(i)
                except OSError as error:
                    logger.error(f"Błąd podczas usuwania pliku {i}: {error}")

    with open(music_metadata_path, "w", encoding="utf-8") as f:
        json.dump(songs_new, f, indent=2, ensure_ascii=False)

    print("Finished successfully!!")

    # for i in range(0,3):
    #     print(songs_new[i].name)
    #     print(songs_new[i].lyrics,"\n\n")
if __name__ == "__main__":
    main()

