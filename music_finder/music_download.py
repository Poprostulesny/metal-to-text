from spotdl import Spotdl
import shutil
import os
import json
from spotdl.types.options import DownloaderOptions
import music_utils as ut
import config as cf
import logging
from progress.bar import IncrementalBar
url = cf.get_music_url()
model = cf.get_model()
location_dir = r"C:\Users\jarek\Documents\programowanie\metal-to-text\music\\"
ollama_url = cf.get_ollama_url()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

shutil.rmtree(location_dir,ignore_errors=True)
os.mkdir(location_dir)
downloader_options: DownloaderOptions = {
    # where and how files should be saved
    "audio_providers": ["youtube-music"],
    "lyrics_providers": ["genius", "azlyrics", "musixmatch"],
    "genius_token": cf.get_genius_token(),
    "playlist_numbering": False,
    "playlist_retain_track_cover": False,
    "scan_for_songs": False,
    "m3u": None,
    "output": location_dir+"{artists} - {title}.{output-ext}",
    "overwrite": "skip",
    "search_query": None,
    "ffmpeg": "ffmpeg",
    "bitrate": "256k",
    "ffmpeg_args": None,
    "format": "wav",
    "save_file": "spotdl_cache.txt",
    "filter_results": True,
    "album_type": None,
    "threads": 8,
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
    "save_errors": None,
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
songs = spotdl.search([url])
songs = spotdl.download_songs(songs)

songs_new = list()
songs_bad =list()
os.system('cls' if os.name == 'nt' else 'clear')
# print(songs[1])
progress_bar = IncrementalBar('Downloading music', max=len(songs))
for song, path in songs:
    if song.lyrics!=None and path != None:
        song.lyrics = ut.sanitized_lyrics(song.lyrics,ollama_url,model,message)
        songs_new.append({
            "path":   str(path),   
            "lyrics": song.lyrics,
            "artist":song.artist,
            "genre":song.genres,
        })
        progress_bar.next()
    else:
        songs_bad.append(str(path))
progress_bar.finish()


if len(songs_bad)!=0:
    for i in songs_bad:
        try:
            os.remove(i)
        except OSError as error:
            logger.error(f"Błąd podczas usuwania pliku {i}: {error}")

with open(location_dir+"music.json", "w", encoding="utf-8") as f:
    json.dump(songs_new, f, indent=2, ensure_ascii=False)

print("Finished successfully!!")

# for i in range(0,3):
#     print(songs_new[i].name)
#     print(songs_new[i].lyrics,"\n\n")





