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