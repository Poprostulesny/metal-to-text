# metal-to-text
My try at using Nvidia Parakeet to decode the gibberish from metal songs

<<<<<<< HEAD
Little guide:
1. First update config.py so that the directories are correct for you.
2. Preprocessing: 
   1.   To use it we first have to download our music in .wav format. We can do it through spotify(file "music_download.py") or provide our own(however then you would have to create your own .json file with lyrics, authors and genre).
      2. Second step is to "clean up" the audio - we strip background music from the .wav file using the "music_preprocess.py" file. On this step we also resample our audio to 16000Hz sampling rate and make sure that it is in mono format.
   3. Third step is splitting our data set into test, train and valid also done by our music_preprocess.py file. If you use my files from start to finish everything should work seamlessly, you just give the playlist into the config and off you go, just start each program.
3. Updating the tokenizer(unnecessary, however if you want to train it on your own then quite necessary):
   1.   To do that we use the "processing_tokenizer.py" file kindly supplied by Nvidia NEMO team with the given commands:
      1.

Step-by-step usage (PL):

1) Wymagania wstępne
   - Zainstaluj zależności: uruchom instalację z pliku [requirements.txt](requirements.txt).
   - Zainstaluj FFmpeg i upewnij się, że jest w `PATH` (wymagane przez SpotDL).
   - Uruchom lokalnie Ollama i pobierz model wskazany w [config.py](config.py) (`get_model()`), domyślnie `gemma2:2b`.
   - (Opcjonalnie/GPU) Skonfiguruj środowisko CUDA; trening w [model/config.yaml](model/config.yaml) używa `accelerator: gpu` i strategii DeepSpeed.

   Przykładowe komendy (PowerShell):

   ```powershell
   # instalacja zależności
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   # sprawdzenie Ollama (lokalny LLM do czyszczenia liryki)
   # pobierz model, np. gemma2:2b
   ollama pull gemma2:2b
   ollama serve
   ```

2) Konfiguracja
   - Otwórz [config.py](config.py) i ustaw:
     - `get_music_path()` — katalog roboczy z muzyką (na Windows np. `C:/Users/.../metal-to-text/music`).
     - `get_data_path()` — katalog na manifesty danych (np. `./data/`).
     - `get_music_url()` — URL playlisty Spotify.
     - `get_ollama_url()` — adres serwera Ollama (domyślnie `http://127.0.0.1:11434`).
     - `get_model()` — nazwa modelu Ollama do sanitizacji tekstu.

3) Pobieranie muzyki i liryki
   - Uruchom pobieranie i czyszczenie liryki: [music_finder/music_download.py](music_finder/music_download.py).

   ```powershell
   python music_finder\music_download.py
   ```

   Efekt: powstanie plik [music/music.json](music/music.json) z polami `path`, `lyrics`, `artist`, `genre`.

   Uwaga: skrypt używa SpotDL. Jeśli potrzebujesz własnych kredencjałów klienta, skonfiguruj je zamiast wartości przykładowych i nie publikuj sekretów.

4) Separacja wokali i przygotowanie danych
   - Uruchom przygotowanie zestawów danych: [music_finder/music_preprocess.py](music_finder/music_preprocess.py).

   ```powershell
   python music_finder\music_preprocess.py
   ```

   Co się dzieje:
   - W [music_finder/music_utils.py](music_finder/music_utils.py) wykonywana jest separacja wokalu (ensemble MDX + Demucs), resampling do 16 kHz mono, normalizacja i zapis do `music_finder/final_music`.
   - Tworzone są manifesty JSONL w [data/](data) — `train_data_8.jsonl`, `valid_data_1.jsonl`, `test_data_1.jsonl` z polami `audio_filepath`, `text`, `duration`, `genre`, `artist`.

5) (Opcjonalnie) Aktualizacja tokenizera
   - Opcja A (prosta): uruchom [parakeet_tokenizer_update.py](parakeet_tokenizer_update.py). Ten skrypt zbuduje tokenizator SentencePiece z treści `text` w manifestach.

   ```powershell
   python parakeet_tokenizer_update.py
   ```

   - Opcja B (pełny skrypt NVIDIA): użyj [proccessing_tokenizer.py](proccessing_tokenizer.py) z argumentami.

   ```powershell
   python proccessing_tokenizer.py \
     --manifest "./data/train_data_8.jsonl,./data/valid_data_1.jsonl,./data/test_data_1.jsonl" \
     --data_root "./model" \
     --vocab_size 2048 \
     --tokenizer spe \
     --spe_type bpe \
     --log
   ```

   Następnie zaktualizuj w [model/config.yaml](model/config.yaml) pole `model.tokenizer.dir` tak, aby wskazywało folder wygenerowanego tokenizera (np. `./model/tokenizer_spe_bpe_v2048`).

6) Konfiguracja treningu
   - W [model/config.yaml](model/config.yaml) ustaw:
     - `init_from_nemo_model` — ścieżka do bazowego modelu `.nemo` (np. Parakeet RNNT 0.6b).
     - `model.train_ds/validation_ds/test_ds.manifest_filepath` — wskaż manifesty w Twoich lokalnych ścieżkach na Windows.
     - Jeśli zaktualizowałeś tokenizer — `model.tokenizer.update_tokenizer: true` oraz `model.tokenizer.dir` na folder tokenizera.
   - Jeżeli nie używasz GPU lub DeepSpeed, rozważ zmianę `trainer.accelerator: cpu` i `trainer.strategy: null`.

7) Trening
   - Uruchom trening skryptem [parakeet_train.py](parakeet_train.py) (Hydra pobierze konfigurację z [model/config.yaml](model/config.yaml)).

   ```powershell
   python parakeet_train.py
   ```

   Podczas treningu logi i checkpointy będą zarządzane przez `exp_manager` zgodnie z ustawieniami w [model/config.yaml](model/config.yaml).

8) Walidacja/inferencja (uwaga)
   - Repozytorium nie zawiera kompletnego skryptu inferencji. Możesz odtworzyć model NeMo i użyć metod `ASRModel` do transkrypcji własnych plików audio. Przykładowy szkic znajduje się w [nemo_training.py](nemo_training.py) (wymaga dopracowania).

Uwagi:
- Ścieżki w konfiguracji przykładowej wskazują Linux (`/home/mateusz/...`). Na Windows zaktualizuj je do swoich lokalnych katalogów.
- Separacja wokali i trening są kosztowne obliczeniowo — preferuj środowisko z GPU.
- Nie publikuj kluczy i sekretów (Spotify/Genius). Przechowuj je lokalnie.
=======
What this project does:
This project is meant to build a small pipeline for metal lyrics transcription. The idea is:
1. download songs and lyrics from a spotify playlist,
2. clean the lyrics a little bit with ollama,
3. isolate vocals from the full song,
4. resample everything to 16000 Hz mono,
5. create NeMo jsonl manifests for train/valid/test,
6. optionally create or update tokenizer,
7. fine tune Nvidia Parakeet / NeMo ASR model on this data.

What files are the important ones:
1. `music_finder/music_download.py` - downloads songs with `spotdl`, saves `.wav` files and `music.json`.
2. `music_finder/music_preprocess.py` - runs vocal separation, resampling and dataset split.
3. `music_finder/music_utils.py` - helper file for lyrics cleanup, vocal ensemble and duration counting.
4. `parakeet_tokenizer_update.py` - creates tokenizer from produced manifests.
5. `tokenizer_extract.py` - extracts tokenizer from `.nemo` model if you want to reuse base tokenizer.
6. `parakeet_train.py` - main training script.
7. `model/config.yaml` - NeMo training config with paths to manifests, tokenizer and base model.

Before start:
1. First update `project_config.py` so that the directories are correct for you. This file is now the main source of truth for the python part of the pipeline.
2. `model/config.yaml` still has its own paths for NeMo training, so if you change training paths later you should keep that file in sync as well.
3. Install `ffmpeg`, because `spotdl` uses it for audio conversion.
4. If you want the lyrics cleanup step, make sure local `ollama` is running and that the model from config exists on your machine.
5. Prepare 2 separate virtual envs, because preprocessing and training use incompatible `numpy` versions.
6. The commands below assume you run them from project root directory.

How to create envs:
1. Data env:
```powershell
py -3.11 -m venv .venv-data
.\.venv-data\Scripts\python -m pip install -U pip setuptools wheel
.\.venv-data\Scripts\python -m pip install -r requirements-data.txt
```
2. Training env:
```powershell
py -3.11 -m venv .venv-train
.\.venv-train\Scripts\python -m pip install -U pip setuptools wheel
.\.venv-train\Scripts\python -m pip install -r requirements-train.txt
```

How the project works step by step:
1. Download music:
Use `music_finder/music_download.py` with `.venv-data`.
This script reads spotify playlist url, downloads songs as `.wav`, tries to fetch lyrics and saves metadata to `music.json`.

Run:
```powershell
.\.venv-data\Scripts\python.exe .\music_finder\music_download.py
```

Output:
1. downloaded wav files,
2. `music.json` with path, lyrics, artist and genre.

2. Preprocess music and build manifests:
Use `music_finder/music_preprocess.py` with `.venv-data`.
This script reads `music.json`, runs 2 vocal separation models, averages them in frequency domain, resamples audio to 16000 Hz mono and then creates train / test / valid jsonl manifests for NeMo.

Run:
```powershell
.\.venv-data\Scripts\python.exe .\music_finder\music_preprocess.py
```

Output:
1. `final_music/` with processed vocals,
2. `test_data_1.jsonl`,
3. `train_data_8.jsonl`,
4. `valid_data_1.jsonl`.

3. Update tokenizer (optional, but for custom training very useful):
Use `parakeet_tokenizer_update.py` with `.venv-train`.
This script reads the manifests and creates a sentencepiece tokenizer in `./model/...`.

Run:
```powershell
.\.venv-train\Scripts\python.exe .\parakeet_tokenizer_update.py
```

After this step you should check `model/config.yaml` and make sure `model.tokenizer.dir` points to the tokenizer directory that was actually created.

4. Extract tokenizer from base model instead of training new one (optional alternative):
Use `tokenizer_extract.py` with `.venv-train`.
This is useful if you want to keep original tokenizer from `.nemo` model.

Run:
```powershell
.\.venv-train\Scripts\python.exe .\tokenizer_extract.py
```

5. Train the model:
Use `parakeet_train.py` with `.venv-train`.
This script loads base NeMo model, loads manifests from `model/config.yaml`, optionally updates tokenizer and starts fine tuning.

Run:
```powershell
.\.venv-train\Scripts\python.exe .\parakeet_train.py
```

Important note:
This training config is set for GPU training and uses deepspeed / bf16 / distributed init. It is not a light cpu script. You should treat it as a proper training run, not as a simple demo script.

Recommended order of running files:
1. update `project_config.py`
2. check `model/config.yaml`
3. create `.venv-data`
4. run `music_finder/music_download.py`
5. run `music_finder/music_preprocess.py`
6. create `.venv-train`
7. run `parakeet_tokenizer_update.py` or `tokenizer_extract.py`
8. run `parakeet_train.py`

Which venv to use with which file:
1. `.venv-data`:
`music_finder/music_download.py`
`music_finder/music_preprocess.py`
2. `.venv-train`:
`parakeet_tokenizer_update.py`
`tokenizer_extract.py`
`parakeet_train.py`
`parakeet test.py`

Extra notes:
1. `parakeet test.py` is only a quick manual test file with hardcoded path, not a full evaluation pipeline.
2. `nemo_training.py` does not seem to be part of the main flow right now.
3. `requirements.txt` is now only an info file, because data and training environments cannot be installed together in one env.
4. `requirements-ui.txt` is outside the main flow. `gradio` is not used in the current training pipeline.
5. The python scripts now read shared paths and runtime values from `project_config.py`.
>>>>>>> efcaadb456902c4c57169a61760b7ef640865e32
