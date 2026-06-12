# Import libraries for AI processing, audio manipulation, and file operations.
from ollama import Client, Message  # Import Message class
import json
import numpy as np
from numpy import ndarray, dtype
from typing import Any
import soundfile as sf
import os
from pathlib import Path
import shutil
import subprocess
import sys
from audio_separator.separator import Separator
import librosa
import re
import gc
import torch

AUG_LEVELS = {1: (15.0, 25.0), 2: (5.0, 15.0)}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import project_config as pc

def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def run_separator_worker(
    audio_paths: list[str],
    output_dir: str,
    model_name: str,
    chunk_size: int = 25,
) -> list[str]:
    worker_script = PROJECT_ROOT / "music_finder" / "separation_worker.py"
    worker_dir = Path(output_dir) / "worker_manifests"
    worker_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    chunk_total = (len(audio_paths) + chunk_size - 1) // chunk_size

    for chunk_number, (start, chunk_paths) in enumerate(chunks(audio_paths, chunk_size), start=1):
        jobs = [
            {"index": start + offset + 1, "total": len(audio_paths), "input": path}
            for offset, path in enumerate(chunk_paths)
        ]
        input_json = worker_dir / f"{model_name}_{start}_jobs.json"
        output_json = worker_dir / f"{model_name}_{start}_results.json"

        with open(input_json, "w", encoding="utf-8") as file:
            json.dump(jobs, file, ensure_ascii=True, indent=2)

        print(
            f"{model_name}: starting worker chunk {chunk_number}/{chunk_total} "
            f"for songs {start + 1}-{start + len(chunk_paths)}",
            flush=True,
        )

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        completed = subprocess.run(
            [
                sys.executable,
                str(worker_script),
                "--input-json",
                str(input_json),
                "--output-json",
                str(output_json),
                "--output-dir",
                output_dir,
                "--model",
                model_name,
                "--chunk-start",
                str(chunk_number),
                "--chunk-total",
                str(chunk_total),
            ],
            check=False,
            env=env,
        )

        if not output_json.exists():
            raise RuntimeError(f"{model_name} worker failed before writing result manifest: {output_json}")

        with open(output_json, "r", encoding="utf-8") as file:
            results = json.load(file)

        failed = [item for item in results if not item["ok"]]
        if failed:
            print(failed[0]["error"], flush=True)
            raise RuntimeError(f"{model_name} worker failed for input: {failed[0]['input']}")

        if completed.returncode != 0:
            raise RuntimeError(f"{model_name} worker exited with code {completed.returncode}")

        all_results.extend(results)

    all_results.sort(key=lambda item: item["index"])
    output_paths = []
    for item in all_results:
        output_paths.extend(item["outputs"])

    return output_paths
# Process lyrics using an AI model to sanitize and format them.
def sanitized_lyrics(lyrics, ollama_url, model, message) -> str | None:
    # Initialize the client with the provided URL.
    client = Client(host=ollama_url)

    # Create Message objects instead of raw dictionaries.
    system_msg = Message(role='system', content='You are a part of a program...')
    user_msg = Message(role='user', content=message + lyrics)

    # Call chat API with the prepared messages.
    response = client.chat(
        model=model,
        messages=[system_msg, user_msg],
        stream=False,
    )
    # Remove newlines from the response for cleaner output.
    content = response.message.content
    if not content:
        return None
    content = content.replace('\n', ' ')
    # content = re.sub(r"[^a-zA-Z.!0-9]", ' ', content)
    content = re.sub(r"\s+", ' ', content)
    return content

# Extract the filename from a full path by getting characters after the last slash or backslash.
def filename(path) -> str:
    name=""
    n = len(path)-1
    # Iterate backwards through the path until a directory separator is found.
    while n>=0 and path[n]!='\\' and path[n]!='/':
        name=path[n]+name
        n-=1
    # name = name.replace(' ', '_')
    return name


def separator_output_path(base_dir: str, path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)

    if candidate.exists():
        return str(candidate)

    joined = Path(base_dir) / candidate
    return str(joined)

def write_ensembled(y1, y2, sr, n_fft, hop_length) -> ndarray[Any, dtype[Any]]:
     # Convert audio to frequency domain using Short-Time Fourier Transform.
    S1 = librosa.stft(y1, n_fft=n_fft, hop_length=hop_length)
    S2 = librosa.stft(y2, n_fft=n_fft, hop_length=hop_length)

    # Create ensemble by averaging the magnitudes from both models.
    mag_avg = 0.5 * (np.abs(S1) + np.abs(S2))

    # Use phase information from the first model for reconstruction.
    phase = np.angle(S1)
    S_avg = mag_avg * np.exp(1j * phase)

    # Convert back to time domain using inverse STFT.
    y_avg = librosa.istft(S_avg, hop_length=hop_length)
    y_avg = librosa.resample(y_avg,orig_sr= sr, target_sr=16000)

    # Find the peak amplitude for normalization.
    peak = np.max(np.abs(y_avg))

    # Normalize audio if peak exceeds 1.0 to prevent clipping.
    if peak > 1.0:
        y_avg = y_avg / peak
    return y_avg;



def active_rms(y, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    active = rms[rms > 0.1 * rms.max()]
    return float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0

def mix_at_snr(y_voc, y_inst, snr_db):
    rms_v = active_rms(y_voc)
    rms_i = active_rms(y_inst)
    if rms_i == 0.0 or rms_v == 0.0:
        return y_voc
    gain = rms_v / (rms_i * 10 ** (snr_db / 20))
    n = min(len(y_voc), len(y_inst))
    y = y_voc[:n] + gain * y_inst[:n]
    peak = np.max(np.abs(y))
    return y / peak if peak > 1.0 else y


# Process music files to extract and enhance vocals using multiple models.
def model(music_lyric_dict, memory_threshold, tmp_dir=None, remove_tmp_files=True):
    if tmp_dir is None:
        tmp_dir = pc.get_tmp_path()
    final_music_dir = pc.get_final_music_path()
    audio_paths = music_lyric_dict['audio_filepath']
    total_songs = len(audio_paths)

    # Keep existing processed stems so preprocessing can run incrementally.
    os.makedirs(final_music_dir, exist_ok=True)

    # Create or recreate the temporary directory for processing.
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)

    # Initialize the audio separator with configuration for vocal extraction.
    sep = Separator(
        output_dir=tmp_dir,
        output_format='WAV',
        output_single_stem=None,
        use_autocast=True,
        demucs_params={
            "segment_size": 45,
            "shifts": 1,
            "overlap": 0.1,
            "segments_enabled": True,
        },
        
    )

    # Load the first vocal separation model and process all audio files.
    out_paths_mdx_vocals:list[str] = []
    out_paths_mdx_instrumentals:list[str] = []
    out_paths_mdx:list[str]=[]
    sep.load_model("vocals_mel_band_roformer.ckpt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    for index, path in enumerate(audio_paths, start=1):
        print(f"RoFormer vocals: processing song {index}/{total_songs}: {filename(path)}", flush=True)
        out_paths_mdx.extend(sep.separate(path,   
                                          custom_output_names={
                                            "Vocals": f"{Path(path).stem}_mdx_vocals",
                                            "Instrumental": f"{Path(path).stem}_mdx_instrumental",
                                            }))
        if torch.cuda.memory_allocated() > total_memory*memory_threshold:
            del sep.model_instance
            sep.model_instance=None
            cuda_cleanup()
            sep.load_model("vocals_mel_band_roformer.ckpt")
    for output_path in out_paths_mdx:
        if output_path.endswith("_mdx_vocals.wav"):
            out_paths_mdx_vocals.append(output_path)
        elif output_path.endswith("_mdx_instrumental.wav"):
            out_paths_mdx_instrumentals.append(output_path)
    # Load the second vocal separation model for ensemble learning.
    del sep
    cuda_cleanup()
    # out_paths_demucs = run_separator_worker(
    #     audio_paths=audio_paths,
    #     output_dir=tmp_dir,
    #     model_name="htdemucs_ft.yaml",
    #     chunk_size=25,
    # )



    # # Validate that all output paths match in length to ensure proper processing.
    if len(out_paths_mdx_vocals) != len(out_paths_mdx_instrumentals):
        raise RuntimeError("Vocal/instrumental stem count mismatch!")
    # Define FFT parameters for spectral processing.
    n_fft = 2048
    hop_length = 512

    # Initialize list to store final processed audio paths.
    final_paths = list()
    rng = np.random.default_rng()
    for i in range(len(out_paths_mdx_vocals)):
        # Construct full paths to the separated audio files.
        out_paths_mdx_vocals[i] = separator_output_path(tmp_dir, out_paths_mdx_vocals[i])
        out_paths_mdx_instrumentals[i]=separator_output_path(tmp_dir, out_paths_mdx_instrumentals[i])
    #    out_paths_demucs[i] = separator_output_path(tmp_dir, out_paths_demucs[i])
        # Load audio data from both models.
        y1, sr = librosa.load(out_paths_mdx_vocals[i], mono=True, sr=16000)
        y_inst, srinst = librosa.load(out_paths_mdx_instrumentals[i], mono=True, sr=16000)

        # Verify sampling rates match between models.
        # if sr != sr2:
        #     print("Error with data, Sampling rate mismatch!", out_paths_mdx[i], out_paths_demucs[i])
        #     quit()
        
        # y_avg = write_ensembled(y1,y2, n_fft, hop_length);
        y_avg = y1;
        # Generate output filename and path for the processed audio.
        input_filename = filename(music_lyric_dict['audio_filepath'][i])
        out_path = str(Path(final_music_dir) / input_filename)
        final_paths.append(out_path)
        
        y_orig, _ = librosa.load( music_lyric_dict['audio_filepath'][i], mono=True, sr=16000)
         # Save the processed audio to disk.
        sf.write(out_path, y_avg, samplerate=16000,format='WAV', )
        sf.write(str(Path(final_music_dir) / f"{Path(input_filename).stem}-3.wav"), y_orig, 16000, format='WAV')
        print(f"Wrote file: {out_path}")
        music_lyric_dict['audio_filepath'][i] = out_path
       
        for level, (lo, hi) in AUG_LEVELS.items():
            snr = float(rng.uniform(lo, hi))
            y_mix = mix_at_snr(y_avg, y_inst, snr)
            aug_path = str(Path(final_music_dir) / f"{Path(input_filename).stem}-{level}.wav")
            sf.write(aug_path, y_mix, samplerate=16000, format='WAV')
            print(f"Wrote aug level {level} (SNR {snr:.1f} dB): {aug_path}")
        # Log the successful processing of the file.
       

    # Clean up temporary directory after processing is complete.
    if remove_tmp_files==True:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    # Return the updated dictionary with paths to processed audio files.
    return music_lyric_dict

def audio_length(path):
    y, sr = librosa.load(path)
    return librosa.get_duration(y=y, sr=sr)
