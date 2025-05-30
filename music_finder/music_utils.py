from ollama import Client, Message  # Import Message class
# import numpy as np
# import soundfile as sf
# import librosa
# import os
# import shutil
# from audio_separator.separator import Separator
#


def sanitized_lyrics(lyrics, ollama_url, model, message) -> str:
    client = Client(host=ollama_url)

    # Create Message objects instead of raw dictionaries
    system_msg = Message(role='system', content='You are a part of a program...')
    user_msg = Message(role='user', content=message + lyrics)

    # Call chat directly with proper message objects
    response = client.chat(
        model=model,
        messages=[system_msg, user_msg]
    )
    response.message.content= response.message.content.replace('\n', ' ')

    return response.message.content

def filename(path) -> str:
    name=""
    n = len(path)-1
    while n>=0 and path[n]!='\\' and path[n]!='/':
        name=path[n]+name
        n-=1
    # name = name.replace(' ', '_')
    return name

# def model(music_lyric_dict, tmp_dir="./tmp",):
#     if not os.path.isdir("./final_music"):
#         os.makedirs("./final_music")
#     else:
#         shutil.rmtree("./final_music", ignore_errors=True)
#         os.makedirs("./final_music")
#
#     if not os.path.isdir(tmp_dir):
#         os.makedirs(tmp_dir)
#     else:
#         shutil.rmtree(tmp_dir, ignore_errors=True)
#         os.makedirs(tmp_dir)
#
#     sep = Separator(
#         output_dir=tmp_dir,
#         output_format='WAV',
#         output_single_stem="Vocals",
#     )
#     # processing
#     sep.load_model("vocals_mel_band_roformer.ckpt")
#     out_paths_mdx = sep.separate(music_lyric_dict['path'])
#
#     # doubling for the other model
#     #
#     sep.load_model("htdemucs_ft.yaml")
#     out_paths_demucs =sep.separate(music_lyric_dict['path'])
#
#
#     # ensuring data is ok
#     if type(out_paths_demucs) is str:
#         out_paths_demucs = [out_paths_demucs]
#         out_paths_mdx = [out_paths_mdx]
#
#     if len(out_paths_demucs)!=len(out_paths_mdx) or len(out_paths_demucs)!=len(music_lyric_dict['path']):
#         print("Length of output paths and demucs paths do not match!")
#         quit()
#     n_fft = 2048
#     hop_length = 512
#
#     final_paths = list()
#     for i in range(len(out_paths_mdx)):
#         out_paths_mdx[i]="./tmp/"+out_paths_mdx[i]
#         out_paths_demucs[i]="./tmp/"+out_paths_demucs[i]
#         y1, sr = librosa.load(out_paths_mdx[i])
#         y2, sr2 = librosa.load(out_paths_demucs[i])
#         if sr != sr2:
#             print("Error with data, Sampling rate mismatch!", out_paths_mdx[i], out_paths_demucs[i])
#             quit()
#
#
#         S1 = librosa.stft(y1, n_fft=n_fft, hop_length=hop_length)
#         S2 = librosa.stft(y2, n_fft=n_fft, hop_length=hop_length)
#         # 3) Average magnitudes
#         mag_avg = 0.5 * (np.abs(S1) + np.abs(S2))
#
#         # Reuse the phase from the first model
#         phase = np.angle(S1)
#         S_avg = mag_avg * np.exp(1j * phase)
#
#         y_avg = librosa.istft(S_avg, hop_length=hop_length)
#         peak = np.max(np.abs(y_avg))
#
#         if peak > 1.0:
#             y_avg = y_avg / peak
#
#         input_filename = filename(music_lyric_dict['path'][i])
#         out_path =r"./final_music/" + input_filename
#         final_paths.append(out_path)
#         music_lyric_dict['path'][i] = out_path
#         sf.write(out_path, y_avg, sr,format='WAV')
#         print(f"Wrote averaged-spec ensemble: {out_path}")
#     shutil.rmtree(tmp_dir, ignore_errors=True)
#     return music_lyric_dict