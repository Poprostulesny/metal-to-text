# metal-to-text
My try at using Nvidia Parakeet to decode the gibberish from metal songs

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