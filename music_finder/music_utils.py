# Import libraries for AI processing, audio manipulation, and file operations.
from ollama import Client, Message  # Import Message class
import numpy as np
import soundfile as sf
import librosa
import os
import shutil
from audio_separator.separator import Separator
import librosa
import config as cf


# Process lyrics using an AI model to sanitize and format them.
def sanitized_lyrics(lyrics, ollama_url, model, message) -> str:
    # Initialize the client with the provided URL.
    client = Client(host=ollama_url)

    # Create Message objects instead of raw dictionaries.
    system_msg = Message(role='system', content='You are a part of a program...')
    user_msg = Message(role='user', content=message + lyrics)

    # Call chat API with the prepared messages.
    response = client.chat(
        model=model,
        messages=[system_msg, user_msg]
    )
    # Remove newlines from the response for cleaner output.
    response.message.content= response.message.content.replace('\n', ' ')

    return response.message.content

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

# Process music files to extract and enhance vocals using multiple models.
def model(music_lyric_dict, tmp_dir="./tmp",):

    # Create or recreate the final music output directory.
    if not os.path.isdir("./final_music"):
        os.makedirs("./final_music")
    else:
        shutil.rmtree("./final_music", ignore_errors=True)
        os.makedirs("./final_music")

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
        output_single_stem="Vocals",
    )

    # Load the first vocal separation model and process all audio files.
    sep.load_model("vocals_mel_band_roformer.ckpt")
    out_paths_mdx = sep.separate(music_lyric_dict['audio_filepath'])

    # Load the second vocal separation model for ensemble learning.
    sep.load_model("htdemucs_ft.yaml")
    out_paths_demucs =sep.separate(music_lyric_dict['audio_filepath'])

    # Convert single string outputs to lists for consistent processing.
    if type(out_paths_demucs) is str:
        out_paths_demucs = [out_paths_demucs]
        out_paths_mdx = [out_paths_mdx]

    # Validate that all output paths match in length to ensure proper processing.
    if len(out_paths_demucs)!=len(out_paths_mdx) or len(out_paths_demucs)!=len(music_lyric_dict['audio_filepath']):
        print("Length of output paths and demucs paths do not match!")
        quit()

    # Define FFT parameters for spectral processing.
    n_fft = 2048
    hop_length = 512

    # Initialize list to store final processed audio paths.
    final_paths = list()
    for i in range(len(out_paths_mdx)):
        # Construct full paths to the separated audio files.
        out_paths_mdx[i]="./tmp/"+out_paths_mdx[i]
        out_paths_demucs[i]="./tmp/"+out_paths_demucs[i]
        # Load audio data from both models.
        y1, sr = librosa.load(out_paths_mdx[i])
        y2, sr2 = librosa.load(out_paths_demucs[i])

        # Verify sampling rates match between models.
        if sr != sr2:
            print("Error with data, Sampling rate mismatch!", out_paths_mdx[i], out_paths_demucs[i])
            quit()

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

        # Generate output filename and path for the processed audio.
        input_filename = filename(music_lyric_dict['audio_filepath'][i])
        out_path =r"./final_music/" + input_filename
        final_paths.append(out_path)
        music_lyric_dict['audio_filepath'][i] = out_path
        # Save the processed audio to disk.
        sf.write(out_path, y_avg, samplerate=16000,format='WAV')

        # Log the successful processing of the file.
        print(f"Wrote averaged-spec ensemble: {out_path}")

    # Clean up temporary directory after processing is complete.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    # Return the updated dictionary with paths to processed audio files.
    return music_lyric_dict

def audio_length(path):
    y, sr = librosa.load(path)
    return librosa.get_duration(y=y, sr=sr)