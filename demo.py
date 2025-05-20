from transformers import pipeline
import gradio as gr
import config as cf

import librosa
# import ffmpeg
# ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Update if using different path
# ffmpeg.input = lambda *args, **kwargs: ffmpeg.input(*args, **kwargs, cmd=ffmpeg_path)
# Path to the folder that contains config.json, model-*.bin, tokenizer.json, etc.
LOCAL_MODEL_DIR = cf.get_model_path()+"/checkpoint-1500"   # <- change to your path

pipe = pipeline(
    task="automatic-speech-recognition",
    model=LOCAL_MODEL_DIR,              # weights + config
    tokenizer=LOCAL_MODEL_DIR,          # tokenizer files
    feature_extractor=LOCAL_MODEL_DIR,  # feature extractor files
    return_timestamps = True,
)   

def transcribe(audio_path):
    audio,sr = librosa.load(audio_path, sr = 16000)
    return pipe({"raw":audio, "sampling_rate":sr})["text"]

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
    title="Whisper Small EN (offline)",
    description="Realtime EN ASR using a locally stored Whisper-turbo model."
)

iface.launch(share=True)