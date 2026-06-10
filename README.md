# metal-to-text
My try at using Nvidia Parakeet to decode the gibberish from metal songs

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
7. If training package installation fails on Transformer Engine / CUDA builds, see `installing_packages.md`.

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
