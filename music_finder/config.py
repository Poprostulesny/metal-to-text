model_path = './whisper-small-test'
timeTest = 15*60
timeTraining = 2*60*60
data_path =r"C:\Users\jarek\Documents\programowanie\metal-to-text\data"

music_url = "https://open.spotify.com/playlist/6SWqIZS4QsIwgRzmo8y5qv?si=90f2480e6db74476"
model = "gemma2:2b"
genius_token="alXXDbPZtK1m2RrZ8I4k2Hn8Ahsd0Gh_o076HYvcdlBvmc0ULL1H8Z8xRlew5qaG"

ollama_url = "http://127.0.0.1:11434"
def get_ollama_url()-> str:
    return ollama_url
def get_model() -> str:
    return model

def get_music_url()-> str:
    return music_url

def get_data_path()-> str:
    return data_path



def get_time()->(int,int):
    return timeTraining,timeTest

def get_model_path()-> str:
    return model_path

def get_genius_token()-> str:
    return genius_token