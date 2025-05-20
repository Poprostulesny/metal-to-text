from datasets import load_dataset, DatasetDict, Dataset, Audio
import os
import time

import config as cf
t  =time.time()

print("Loading data") 
stream_train = load_dataset(
    "mozilla-foundation/common_voice_11_0", "en", split="train",trust_remote_code=True, streaming=True,
)
stream_test = load_dataset(
    "mozilla-foundation/common_voice_11_0", "en", split="test",trust_remote_code=True, streaming=True,
)
print("Took: " , round(time.time()-t,2), 's')
t=time.time()
print("Shuffling data")

stream_train=stream_train.shuffle(buffer_size=10_000, seed=42)
stream_test=stream_test.shuffle(buffer_size=10_000, seed=22)
print("Took: " , round(time.time()-t,2), 's')
t=time.time()

def train_generator():
    yield from stream_train.take(5000)

def test_generator():
    yield from stream_test.take(1500)


# samples_train = list(stream_train.take(1500))
# samples_test =list(stream_test.take(200))
# dataset_train= Dataset.from_list(samples_train)
# dataset_test = Dataset.from_list(samples_test)
print("Generating datasets")
dataset_train = Dataset.from_generator(train_generator)
dataset_test = Dataset.from_generator(test_generator)
print("Took: " , round(time.time()-t,2))
t=time.time()
print("Resampling audio")
dataset_test  = dataset_test.cast_column("audio", Audio(sampling_rate=16000))
dataset_train  = dataset_train.cast_column("audio", Audio(sampling_rate=16000))
print("Took: " , round(time.time()-t,2), 's')
t=time.time()
out_dir_test = cf.get_out_dir_test()

out_dir_train =cf.get_out_dir_train
if not os.path.isdir(out_dir_test):
    os.makedirs(out_dir_test)
if not os.path.isdir(out_dir_train):
    os.makedirs(out_dir_train)
print("Saving data")

dataset_train.save_to_disk(out_dir_train)
dataset_test.save_to_disk(out_dir_test)
print("Took: " , round(time.time()-t,2), 's')
