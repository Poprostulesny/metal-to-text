from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
import threading
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from transformers import WhisperFeatureExtractor as WFE
from transformers import WhisperTokenizer as WT
from transformers import WhisperProcessor as WP
import evaluate
import time
import random
import config as cf
### config of the model
feature_extractor = WFE.from_pretrained("openai/whisper-large-v3-turbo")
tokenizer = WT.from_pretrained("openai/whisper-large-v3-turbo", language="English", task='transcribe')
processor = WP.from_pretrained("openai/whisper-large-v3-turbo", language="English", task="transcribe")
metric = evaluate.load("wer")
class Counter:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()
    def increment(self, n=1):
        with self._lock:
            self.value += n
    def get(self):
        with self._lock:
            return self.value
        
def import_train(tTarget, stream,counter):
    acc=0.0
    selected_train=[]
    for example in stream["train"]:
        dur=len(example["audio"]["array"])/ example["audio"]["sampling_rate"]
        if(acc+dur>tTarget):
            break
        acc+=dur
        counter.increment(dur)

        selected_train.append(example)
    return selected_train


def import_test(tTarget, stream,counter):
    acc=0.0
    selected_test=[]
    for example in stream["test"]:
        dur=len(example["audio"]["array"])/ example["audio"]["sampling_rate"]
        if(acc+dur>tTarget):
            break
        acc+=dur
        counter.increment(dur)
        selected_test.append(example)

    return selected_test

def prepare_dataset(batch):
   
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
    
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def build_stream():
    stream = DatasetDict()
    stream["train"] = load_from_disk(cf.get_out_dir_train()).shuffle( seed=int(random.random()))
    stream["test"] =load_from_disk(cf.get_out_dir_test()).shuffle( seed=int(random.random()))
    # print("Stream before changing format:\n")
    # print(stream)
    # for split in ("train", "test"):
    #     stream[split].set_format(type="numpy", columns=["audio"])
    return stream

def get_files(stream):
    tTarget_train, tTarget_test = cf.get_time()
    tsum=tTarget_test+tTarget_train
    with ThreadPoolExecutor(max_workers=2) as exec:
        counter = Counter()
        futures ={
            "test":exec.submit(import_test, tTarget_test, stream, counter),
            "train":exec.submit(import_train, tTarget_train, stream,counter)
        }
        def monitor():
            while any(not fut.done() for fut in futures.values()):
                cnt = counter.get()
                sys.stdout.write(f"\rProgress: {round(cnt/tsum*100,2)} %...")
                sys.stdout.flush()
                time.sleep(0.2)
            # Final update
            print(f"\rProgress: 100% (done)")

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        
        for fut in as_completed(futures.values()):
            for name, f in futures.items():
                if f is fut:
                    stream[name] = fut.result()
                    print(f"Stream[{name}] loaded.")
                    break
        monitor_thread.join()
    return stream